import { EXERCISE_BY_ID } from '../data/exercises';
import type { Block, Step, WorkoutItem } from '../types';

export const PREPARE_SEC = 10;
export const SWITCH_SIDE_SEC = 5;
export const BLOCK_TRANSITION_SEC = 15;

// 单侧（或不分侧）一次动作的秒数
export function workSeconds(item: WorkoutItem): number {
    if (item.seconds !== undefined)
        return item.seconds;
    const ex = EXERCISE_BY_ID[item.exerciseId];
    return Math.round((item.reps ?? 0) * ex.secPerRep);
}

function workLabel(item: WorkoutItem): string {
    const ex = EXERCISE_BY_ID[item.exerciseId];
    if (item.seconds !== undefined)
        return `${ex.name} ${item.seconds} 秒`;
    return `${ex.name} ${item.reps} 次`;
}

interface Pending {
    work: Step[];               // 一组动作（单侧动作为左右两步）
    restAfter: number;
}

function expandItem(item: WorkoutItem, blockIndex: number, extra: Partial<Step>): Step[] {
    const ex = EXERCISE_BY_ID[item.exerciseId];
    const base: Step = {
        kind: 'work',
        exerciseId: item.exerciseId,
        seconds: workSeconds(item),
        label: workLabel(item),
        blockIndex,
        reps: item.reps,
        ...extra,
    };
    if (!ex.perSide)
        return [base];
    return [
        { ...base, side: 'left' },
        { ...base, kind: 'rest', seconds: SWITCH_SIDE_SEC, label: '换另一侧', reps: undefined },
        { ...base, side: 'right' },
    ];
}

function blockPendings(block: Block, blockIndex: number): Pending[] {
    const out: Pending[] = [];
    for (let r = 0; r < block.rounds; r++) {
        block.items.forEach((item, i) => {
            for (let s = 0; s < item.sets; s++) {
                const extra: Partial<Step> = {};
                if (item.sets > 1)
                    extra.set = { index: s + 1, count: item.sets };
                if (block.rounds > 1)
                    extra.round = { index: r + 1, count: block.rounds };
                let restAfter = item.restSec;
                const lastInRound = i === block.items.length - 1 && s === item.sets - 1;
                if (lastInRound && r < block.rounds - 1)
                    restAfter = block.restBetweenRoundsSec;
                out.push({ work: expandItem(item, blockIndex, extra), restAfter });
            }
        });
    }
    return out;
}

// 把整节课展开成按秒计的步骤序列。休息步骤预告下一个动作；最后一个动作后直接结束
export function buildTimeline(blocks: Block[]): Step[] {
    const pendings: { p: Pending; blockIndex: number }[] = [];
    blocks.forEach((b, bi) => {
        for (const p of blockPendings(b, bi))
            pendings.push({ p, blockIndex: bi });
    });
    if (pendings.length === 0)
        return [];

    const steps: Step[] = [];
    const first = pendings[0].p.work[0];
    steps.push({
        kind: 'prepare',
        exerciseId: first.exerciseId,
        seconds: PREPARE_SEC,
        label: `准备开始，第一个动作：${EXERCISE_BY_ID[first.exerciseId].name}`,
        blockIndex: 0,
    });
    pendings.forEach(({ p, blockIndex }, i) => {
        steps.push(...p.work);
        const next = pendings[i + 1];
        if (!next)
            return;
        const nextEx = EXERCISE_BY_ID[next.p.work[0].exerciseId];
        const changingBlock = next.blockIndex !== blockIndex;
        const seconds = changingBlock ? BLOCK_TRANSITION_SEC : p.restAfter;
        if (seconds <= 0)
            return;
        steps.push({
            kind: 'rest',
            exerciseId: nextEx.id,
            seconds,
            label: `休息，下一个：${nextEx.name}`,
            blockIndex: next.blockIndex,
        });
    });
    return steps;
}

export function totalSeconds(steps: Step[]): number {
    return steps.reduce((sum, s) => sum + s.seconds, 0);
}

// 给定已进行的秒数，定位当前步骤与该步骤已进行的秒数；超出总时长返回 index = steps.length
export function locate(steps: Step[], elapsedSec: number): { index: number; intoStep: number } {
    let start = 0;
    for (let i = 0; i < steps.length; i++) {
        if (elapsedSec < start + steps[i].seconds)
            return { index: i, intoStep: elapsedSec - start };
        start += steps[i].seconds;
    }
    return { index: steps.length, intoStep: 0 };
}

export function stepStart(steps: Step[], index: number): number {
    return totalSeconds(steps.slice(0, index));
}
