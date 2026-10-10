import { EXERCISE_BY_ID } from '../data/exercises';
import type { Block, Phase, Workout, WorkoutItem } from '../types';
import { buildTimeline, totalSeconds } from './timeline';

// 手动调整训练：加入 / 移除动作。时长始终由跟练时间线重新算出

const FLOW_REST_SEC = 5;
const BLOCK_TITLE: Record<Phase, string> = { warmup: '热身', main: '主训练 · 自选', cooldown: '放松' };

function withEstimate(w: Workout, blocks: Block[]): Workout {
    const kept = blocks.filter((b) => b.items.length > 0);
    return { ...w, blocks: kept, estimatedMin: Math.round(totalSeconds(buildTimeline(kept)) / 60) };
}

// 动作放进哪个阶段：能做主训练就放主训练，否则热身或放松
function phaseFor(exerciseId: string): Phase {
    const phases = EXERCISE_BY_ID[exerciseId].phases;
    if (phases.includes('main'))
        return 'main';
    return phases.includes('warmup') ? 'warmup' : 'cooldown';
}

// 新动作的组数、次数/时长、休息跟随所在区块，保持节奏一致
function itemFor(exerciseId: string, block: Block | undefined): WorkoutItem {
    const ex = EXERCISE_BY_ID[exerciseId];
    const first = block?.items[0];
    const dose = ex.mode === 'time' ? { seconds: ex.defaultSeconds } : { reps: ex.defaultReps };
    if (!block || block.format === 'flow')
        return { exerciseId, sets: 1, ...dose, restSec: FLOW_REST_SEC };
    if (block.format === 'circuit')
        return { exerciseId, sets: 1, seconds: first?.seconds ?? 40, restSec: first?.restSec ?? 20 };
    return { exerciseId, sets: first?.sets ?? 3, ...dose, restSec: first?.restSec ?? 60 };
}

export function addExercise(w: Workout, exerciseId: string): Workout {
    const phase = phaseFor(exerciseId);
    const blocks = w.blocks.map((b) => ({ ...b, items: [...b.items] }));
    let block = blocks.find((b) => b.phase === phase);
    if (!block) {
        block = {
            phase,
            title: BLOCK_TITLE[phase],
            format: phase === 'main' ? 'straight' : 'flow',
            rounds: 1,
            restBetweenRoundsSec: phase === 'main' ? 0 : FLOW_REST_SEC,
            items: [],
        };
        // 热身放最前，放松放最后，主训练放在放松之前
        if (phase === 'warmup')
            blocks.unshift(block);
        else if (phase === 'cooldown')
            blocks.push(block);
        else {
            const cool = blocks.findIndex((b) => b.phase === 'cooldown');
            blocks.splice(cool < 0 ? blocks.length : cool, 0, block);
        }
    }
    block.items.push(itemFor(exerciseId, block.items.length > 0 ? block : undefined));
    return withEstimate(w, blocks);
}

export function removeItem(w: Workout, blockIndex: number, itemIndex: number): Workout {
    const blocks = w.blocks.map((b, bi) => (bi === blockIndex ? { ...b, items: b.items.filter((_, i) => i !== itemIndex) } : b));
    return withEstimate(w, blocks);
}
