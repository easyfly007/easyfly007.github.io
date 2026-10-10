import { useEffect, useRef, useState } from 'react';
import { viewIndexAt } from '../anim/schedule';
import type { Stage } from '../anim/stage';
import type { Gender, Step } from '../types';

interface Props {
    step: Step;
    intoStep: number;
    paused: boolean;
    coach: Gender;
}

// 跟练时的 3D 人偶动作指导：按计时播放，每做完一次（太快的动作每几次）换一个视角。
// three.js 按需加载，不拖慢首页；浏览器不支持 WebGL 时不显示
export function ExerciseAnimation({ step, intoStep, paused, coach }: Props) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [stage, setStage] = useState<Stage | null>(null);
    const [failed, setFailed] = useState(false);
    const [viewLabel, setViewLabel] = useState('');

    // 计时每 0.1 秒更新一次；动画每帧在最近一次计时上往后推，画面更顺
    const clock = useRef({ t: intoStep, at: 0, paused });
    useEffect(() => {
        clock.current = { t: intoStep, at: performance.now(), paused };
    }, [intoStep, paused]);

    useEffect(() => {
        let alive = true;
        let created: Stage | null = null;
        const dark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false;
        import('../anim/stage')
            .then(({ createStage }) => {
                if (!alive || !canvasRef.current)
                    return;
                try {
                    created = createStage(canvasRef.current, dark);
                    setStage(created);
                } catch {
                    setFailed(true);
                }
            })
            .catch(() => setFailed(true));
        return () => {
            alive = false;
            created?.dispose();
        };
    }, []);

    useEffect(() => {
        stage?.setExercise(step.exerciseId, step.side, coach);
    }, [stage, step.exerciseId, step.side, coach]);

    useEffect(() => {
        if (!stage)
            return;
        let raf = 0;
        let last = '';
        const onResize = () => stage.resize();
        window.addEventListener('resize', onResize);
        // 全屏切换等只改变画布大小、不触发窗口 resize 的情况
        const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(onResize);
        if (canvasRef.current)
            observer?.observe(canvasRef.current);
        const loop = () => {
            raf = requestAnimationFrame(loop);
            const c = clock.current;
            const t = c.paused ? c.t : c.t + (performance.now() - c.at) / 1000;
            const label = stage.render(t, viewIndexAt(step.exerciseId, t));
            if (label !== last) {
                last = label;
                setViewLabel(label);
            }
        };
        loop();
        return () => {
            cancelAnimationFrame(raf);
            window.removeEventListener('resize', onResize);
            observer?.disconnect();
        };
    }, [stage, step.exerciseId]);

    if (failed)
        return null;
    return (
        <div className="exercise-video">
            <canvas ref={canvasRef} className="anim-canvas" />
            {viewLabel && <span className="video-view">{viewLabel}</span>}
        </div>
    );
}
