import { EXERCISES } from '../data/exercises';
import type { Block, Conditions, Exercise, Muscle, Phase, RestLevel, Workout, WorkoutItem } from '../types';
import { candidates } from './filter';
import { createRng } from './random';
import { BLOCK_TRANSITION_SEC, PREPARE_SEC, buildTimeline, totalSeconds } from './timeline';

type Rng = () => number;

const FLOW_REST_SEC = 5;
const MAX_FLOW_CYCLES = 3;
const MAX_HOLD_SCALE = 2;
const ROUND_REST_SEC = 60;
// 循环训练 做/休 秒数，按强度 低/中/高
const CIRCUIT_WORK = [0, 30, 40, 45];
const CIRCUIT_REST = [0, 30, 20, 15];
// 常规组 组数与组间休息
const STRAIGHT_SETS = [0, 2, 3, 4];
const STRAIGHT_REST = [0, 60, 75, 90];
const MAX_SETS = 5;
// 休息档位：在默认休息上 少/多 15 秒，最少 10 秒（留出预告下一个动作的时间）
const REST_DELTA: Record<RestLevel, number> = { short: -15, normal: 0, long: 15 };
const MIN_REST_SEC = 10;
const MAX_STRAIGHT_ITEMS = 10;

export function blockSeconds(block: Block): number {
    const steps = buildTimeline([block]);
    if (steps.length === 0)
        return 0;
    return totalSeconds(steps) - PREPARE_SEC;
}

// 主训练的组间/动作间/轮间休息按档位调整；热身、放松、拉伸流的 5 秒衔接不调整
function adjustRest(sec: number, c: Conditions): number {
    return Math.max(MIN_REST_SEC, sec + REST_DELTA[c.rest ?? 'normal']);
}

function preferenceScore(e: Exercise, c: Conditions, phase: Phase): number {
    let score = 0;
    if (phase !== 'main')
        return score;
    if (c.goal === 'fat_loss' && e.category === 'cardio')
        score += 0.6;
    if (c.goal === 'strength' && e.category === 'strength')
        score += 0.4;
    if (c.goal === 'strength' && e.equipment.length > 0)
        score += 0.4;
    if (c.intensity === 3)
        score += (e.intensity / 5) * 0.5;
    if (c.intensity === 1)
        score += ((5 - e.intensity) / 5) * 0.5;
    return score;
}

// 打乱并排序：随机 + 目标偏好 + 优先覆盖还没练到的肌群
function orderPool(pool: Exercise[], c: Conditions, phase: Phase, rng: Rng): Exercise[] {
    const rest = pool.map((e) => ({ e, base: rng() + preferenceScore(e, c, phase) }));
    const covered = new Set<Muscle>();
    const out: Exercise[] = [];
    while (rest.length > 0) {
        let best = 0;
        let bestScore = -Infinity;
        rest.forEach(({ e, base }, i) => {
            const fresh = e.muscles.some((m) => !covered.has(m));
            const score = base + (fresh ? 0.8 : 0);
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        });
        const [picked] = rest.splice(best, 1);
        picked.e.muscles.forEach((m) => covered.add(m));
        out.push(picked.e);
    }
    return out;
}

function defaultItem(e: Exercise, sets: number, restSec: number): WorkoutItem {
    if (e.mode === 'time')
        return { exerciseId: e.id, sets, seconds: e.defaultSeconds, restSec };
    return { exerciseId: e.id, sets, reps: e.defaultReps, restSec };
}

// 计时动作的停留时间按比例拉长（最多 2 倍），取 5 秒整数
function stretchItem(item: WorkoutItem, scale: number): WorkoutItem {
    if (item.seconds === undefined)
        return item;
    return { ...item, seconds: Math.round((item.seconds * scale) / 5) * 5 };
}

// 依次加入动作（只要更接近目标时长）；动作不够时先拉长停留时间，再从头循环
function flowBlock(phase: Phase, title: string, ordered: Exercise[], targetSec: number): Block {
    const block: Block = { phase, title, format: 'flow', rounds: 1, restBetweenRoundsSec: FLOW_REST_SEC, items: [] };
    if (ordered.length === 0)
        return block;
    const oneCycle = blockSeconds({ ...block, items: ordered.map((e) => defaultItem(e, 1, FLOW_REST_SEC)) });
    const scale = Math.min(MAX_HOLD_SCALE, Math.max(1, targetSec / oneCycle));
    for (let i = 0; i < ordered.length * MAX_FLOW_CYCLES; i++) {
        const item = stretchItem(defaultItem(ordered[i % ordered.length], 1, FLOW_REST_SEC), scale);
        const next = { ...block, items: [...block.items, item] };
        const closer = Math.abs(blockSeconds(next) - targetSec) < Math.abs(blockSeconds(block) - targetSec);
        if (block.items.length > 0 && !closer)
            break;
        block.items = next.items;
    }
    return block;
}

function circuitBlock(ordered: Exercise[], c: Conditions, targetSec: number): Block {
    const work = CIRCUIT_WORK[c.intensity];
    const rest = adjustRest(CIRCUIT_REST[c.intensity], c);
    const items = ordered.map((e) => ({ exerciseId: e.id, sets: 1, seconds: work, restSec: rest }));
    // 在 每轮 3–6 个动作 × 1–10 轮 中选最接近目标时长的组合
    let best: Block | undefined;
    let bestDiff = Infinity;
    for (let perRound = Math.min(3, items.length); perRound <= Math.min(6, items.length); perRound++) {
        for (let rounds = 1; rounds <= 10; rounds++) {
            const block: Block = {
                phase: 'main',
                title: '主训练 · 循环',
                format: 'circuit',
                rounds,
                restBetweenRoundsSec: adjustRest(ROUND_REST_SEC, c),
                items: items.slice(0, perRound),
            };
            const diff = Math.abs(blockSeconds(block) - targetSec);
            if (diff < bestDiff) {
                bestDiff = diff;
                best = block;
            }
        }
    }
    return best ?? { phase: 'main', title: '主训练 · 循环', format: 'circuit', rounds: 1, restBetweenRoundsSec: 0, items: [] };
}

function straightBlock(ordered: Exercise[], c: Conditions, targetSec: number): Block {
    const restSec = adjustRest(c.canShower ? STRAIGHT_REST[c.intensity] : 60, c);
    const preferredSets = STRAIGHT_SETS[c.intensity];
    // 在 动作数 × 组数 中选最接近目标时长的组合；误差相近时偏向默认组数
    let best: Block | undefined;
    let bestCost = Infinity;
    for (let n = 1; n <= Math.min(MAX_STRAIGHT_ITEMS, ordered.length); n++) {
        for (let sets = 2; sets <= MAX_SETS; sets++) {
            const block: Block = {
                phase: 'main',
                title: '主训练 · 力量',
                format: 'straight',
                rounds: 1,
                restBetweenRoundsSec: 0,
                items: ordered.slice(0, n).map((e) => defaultItem(e, sets, restSec)),
            };
            const cost = Math.abs(blockSeconds(block) - targetSec) + Math.abs(sets - preferredSets) * 30;
            if (cost < bestCost) {
                bestCost = cost;
                best = block;
            }
        }
    }
    return best ?? { phase: 'main', title: '主训练 · 力量', format: 'straight', rounds: 1, restBetweenRoundsSec: 0, items: [] };
}

function mainPool(c: Conditions): Exercise[] {
    const all = candidates(EXERCISES, c, 'main');
    if (c.goal === 'flexibility') {
        const flex = all.filter((e) => e.category === 'stretch' || e.category === 'mobility');
        if (flex.length > 0)
            return flex;
        return candidates(EXERCISES, c, 'cooldown');
    }
    return all.filter((e) => e.category !== 'stretch');
}

function mainBlock(c: Conditions, rng: Rng, targetSec: number): Block {
    const ordered = orderPool(mainPool(c), c, 'main', rng);
    if (c.goal === 'flexibility')
        return flowBlock('main', '主训练 · 拉伸', ordered, targetSec);
    if (c.goal === 'strength' || !c.canShower)
        return straightBlock(ordered, c, targetSec);
    return circuitBlock(ordered, c, targetSec);
}

export function generateWorkout(c: Conditions, seed: number): Workout {
    const rng = createRng(seed);
    const total = c.durationMin * 60;
    const edgeTarget = Math.max(120, Math.round(total * 0.15));

    const warmPool = orderPool(candidates(EXERCISES, c, 'warmup'), c, 'warmup', rng);
    const warm = flowBlock('warmup', '热身', warmPool, edgeTarget);
    const coolPool = orderPool(candidates(EXERCISES, c, 'cooldown'), c, 'cooldown', rng);
    const cool = flowBlock('cooldown', '放松', coolPool, edgeTarget);

    const overhead = PREPARE_SEC + 2 * BLOCK_TRANSITION_SEC;
    const mainTarget = total - overhead - blockSeconds(warm) - blockSeconds(cool);
    const main = mainBlock(c, rng, mainTarget);

    const blocks = [warm, main, cool].filter((b) => b.items.length > 0);
    const seconds = totalSeconds(buildTimeline(blocks));
    return { seed, conditions: c, blocks, estimatedMin: Math.round(seconds / 60) };
}
