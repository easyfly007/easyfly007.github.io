import { EXERCISE_BY_ID } from '../data/exercises';
import { ANIMATIONS } from './library';

// 动画的播放节奏：一个循环 = 计次动作的一次（secPerRep），计时动作用动画自己的 cycleSec。
// 视角在循环交界处切换（每次动作首尾姿态相同，切换时动作连贯）；
// 一个循环太短时（如开合跳）连续几次再换，保证每个视角至少看 2.5 秒

export type AnimView = 'front' | 'side' | 'full';

export const ANIM_VIEW_LABEL: Record<AnimView, string> = {
    front: '正面',
    side: '侧面',
    full: '全身',
};

const MIN_VIEW_SEC = 2.5;
const DEFAULT_CYCLE_SEC = 2;

export function hasAnimation(exerciseId: string): boolean {
    return exerciseId in ANIMATIONS;
}

export function animCycleSec(exerciseId: string): number {
    const ex = EXERCISE_BY_ID[exerciseId];
    if (ex.mode === 'reps')
        return ex.secPerRep;
    return ANIMATIONS[exerciseId]?.cycleSec ?? DEFAULT_CYCLE_SEC;
}

export function viewIndexAt(exerciseId: string, t: number): number {
    const cycle = animCycleSec(exerciseId);
    const perView = Math.max(1, Math.ceil(MIN_VIEW_SEC / cycle));
    return Math.floor(Math.max(0, t) / (cycle * perView)) % 3;
}
