import { EXERCISE_BY_ID } from '../data/exercises';
import type { Exercise, Step } from '../types';

// 动作视频：每个动作每个视角只存"一次动作"的短片段，跟练时按次拼接，
// 每做完一次换一个视角（正面 → 侧面 → 全身 → 正面 …）。
// 每个片段都从同一个起始姿态开始、回到同一个起始姿态结束，所以在两次动作的交界处切换视角，动作是连贯的
export type ClipView = 'front' | 'side' | 'full';

export const CLIP_VIEWS: ClipView[] = ['front', 'side', 'full'];

export const CLIP_VIEW_LABEL: Record<ClipView, string> = {
    front: '正面',
    side: '侧面',
    full: '全身',
};

// 计时动作（平板支撑、开合跳等）一个片段的时长，也是换视角的间隔
export const TIME_CLIP_SEC = 5;

// 片段文件名：front.mp4；单侧动作为 front-left.mp4 / front-right.mp4
export function clipName(view: ClipView, side?: 'left' | 'right'): string {
    return side ? `${view}-${side}` : view;
}

// 一个片段在跟练中占的秒数：计次动作 = 一次动作的时长；计时动作 = TIME_CLIP_SEC
export function cycleSeconds(ex: Exercise): number {
    return ex.mode === 'reps' ? ex.secPerRep : TIME_CLIP_SEC;
}

export interface ClipPosition {
    cycle: number;              // 当前是这一步里的第几个片段（从 0 开始）
    view: ClipView;
    progress: number;           // 在当前片段里的进度 0..1
}

// 给定跟练步骤和步骤内已进行的秒数，算出该播哪个视角的片段、播到哪里。
// views 是这个动作实际有的视角（缺视角时只在有的里面轮换）
export function clipAt(step: Step, intoStep: number, views: ClipView[] = CLIP_VIEWS): ClipPosition | null {
    if (step.kind !== 'work' || views.length === 0)
        return null;
    const cycleSec = cycleSeconds(EXERCISE_BY_ID[step.exerciseId]);
    const t = Math.max(0, Math.min(intoStep, step.seconds));
    const cycle = Math.min(Math.floor(t / cycleSec), Math.max(0, Math.ceil(step.seconds / cycleSec) - 1));
    const progress = Math.min(1, (t - cycle * cycleSec) / cycleSec);
    return { cycle, view: views[cycle % views.length], progress };
}
