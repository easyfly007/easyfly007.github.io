import { useEffect, useRef } from 'react';
import { EXERCISE_BY_ID } from '../data/exercises';
import { CLIP_VIEW_LABEL, clipAt, cycleSeconds } from '../engine/clips';
import type { Step } from '../types';
import { clipUrls } from '../videos';

interface Props {
    step: Step;
    intoStep: number;
    paused: boolean;
}

// 播放位置与计时的偏差超过这么多秒（按片段自身时间算）就重新对齐
const MAX_DRIFT_SEC = 0.3;

// 浏览器录制的片段常常没有写时长（duration 为 Infinity），跳到末尾一次浏览器才会算出来
function fixUnknownDuration(e: React.SyntheticEvent<HTMLVideoElement>) {
    const v = e.currentTarget;
    if (v.duration !== Infinity)
        return;
    v.addEventListener('durationchange', () => {
        v.currentTime = 0;
    }, { once: true });
    v.currentTime = 1e9;
}

// 按跟练计时拼接动作片段：每个视角一个 <video> 叠在一起，只显示当前视角那个，切换时不黑屏。
// 片段按播放速度缩放到"一次动作"的时长，所以画面始终和计时、计次对齐
export function ExerciseVideo({ step, intoStep, paused }: Props) {
    const clips = clipUrls(step.exerciseId, step.side);
    const pos = clipAt(step, intoStep, clips.map((c) => c.view));
    const refs = useRef<(HTMLVideoElement | null)[]>([]);
    const cycleSec = cycleSeconds(EXERCISE_BY_ID[step.exerciseId]);

    useEffect(() => {
        if (!pos)
            return;
        refs.current.forEach((v, i) => {
            if (!v)
                return;
            if (clips[i]?.view !== pos.view) {
                if (!v.paused)
                    v.pause();
                return;
            }
            if (!Number.isFinite(v.duration) || v.duration <= 0)
                return;
            v.playbackRate = v.duration / cycleSec;
            const target = Math.min(pos.progress * v.duration, v.duration - 0.05);
            if (Math.abs(v.currentTime - target) > MAX_DRIFT_SEC)
                v.currentTime = target;
            if (paused)
                v.pause();
            else if (v.paused)
                void v.play().catch(() => undefined);
        });
    });

    if (!pos)
        return null;
    return (
        <div className="exercise-video">
            {clips.map((c, i) => (
                <video
                    key={c.url}
                    ref={(el) => {
                        refs.current[i] = el;
                    }}
                    src={c.url}
                    muted
                    playsInline
                    preload="auto"
                    onLoadedMetadata={fixUnknownDuration}
                    className={c.view === pos.view ? 'on' : undefined}
                />
            ))}
            <span className="video-view">{CLIP_VIEW_LABEL[pos.view]}</span>
        </div>
    );
}
