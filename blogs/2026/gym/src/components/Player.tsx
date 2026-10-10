import { useEffect, useMemo, useRef, useState } from 'react';
import { EXERCISE_BY_ID } from '../data/exercises';
import { buildTimeline, locate, stepStart, totalSeconds } from '../engine/timeline';
import { formatDuration } from '../labels';
import { beep, speak, stopSpeaking } from '../speech';
import type { Gender, Step, Workout } from '../types';
import { hasAnimation } from '../anim/schedule';
import { clipUrls } from '../videos';
import { ExerciseBodyMap } from './BodyMap';
import { ExerciseAnimation } from './ExerciseAnimation';
import { ExerciseVideo } from './ExerciseVideo';

interface Props {
    workout: Workout;
    coach: Gender;
    // 训练做完或中途结束时调用，传出实际播放过的时间线区间（秒）
    onExit: (played: [number, number][]) => void;
}

function announce(step: Step): string {
    const ex = EXERCISE_BY_ID[step.exerciseId];
    if (step.kind === 'prepare')
        return step.label;
    if (step.kind === 'rest') {
        if (step.label === '换另一侧')
            return '换另一侧';
        return `休息 ${step.seconds} 秒，下一个动作：${ex.name}`;
    }
    const side = step.side === 'left' ? '左侧，' : step.side === 'right' ? '右侧，' : '';
    const amount = step.reps !== undefined ? `${step.reps} 次` : `${step.seconds} 秒`;
    const setInfo = step.set ? `，第 ${step.set.index} 组` : '';
    return `${side}开始，${ex.name} ${amount}${setInfo}`;
}

// 屏幕常亮：跟练过程中不让手机/电脑熄屏
function useWakeLock(active: boolean) {
    useEffect(() => {
        if (!active || !('wakeLock' in navigator))
            return;
        let lock: WakeLockSentinel | null = null;
        navigator.wakeLock.request('screen').then((l) => {
            lock = l;
        }).catch(() => undefined);
        return () => {
            void lock?.release();
        };
    }, [active]);
}

export function Player({ workout, coach, onExit }: Props) {
    const steps = useMemo(() => buildTimeline(workout.blocks), [workout]);
    const total = useMemo(() => totalSeconds(steps), [steps]);

    // 用时间戳计时，避免 setInterval 漂移；暂停时把已用时间存进 baseMs
    const [paused, setPaused] = useState(false);
    const baseMs = useRef(0);
    const startedAt = useRef(performance.now());
    const [elapsedMs, setElapsedMs] = useState(0);

    useEffect(() => {
        if (paused)
            return;
        startedAt.current = performance.now();
        const id = window.setInterval(() => {
            setElapsedMs(baseMs.current + performance.now() - startedAt.current);
        }, 100);
        return () => {
            window.clearInterval(id);
            baseMs.current += performance.now() - startedAt.current;
        };
    }, [paused]);

    const elapsed = Math.min(elapsedMs / 1000, total);
    const { index, intoStep } = locate(steps, elapsed);
    const finished = index >= steps.length;
    const step = steps[index];
    const remaining = step ? Math.ceil(step.seconds - intoStep) : 0;

    useWakeLock(!finished && !paused);

    // 记录实际播放过的区间：跳过时结束当前区间，从跳到的位置开始新区间
    const played = useRef<[number, number][]>([]);
    const segStart = useRef(0);
    function closeSegment(at: number) {
        if (at > segStart.current)
            played.current.push([segStart.current, at]);
    }

    // 进入新步骤时播报
    const lastIndex = useRef(-1);
    useEffect(() => {
        if (index === lastIndex.current)
            return;
        lastIndex.current = index;
        if (finished) {
            speak('训练完成，辛苦了！');
            closeSegment(total);
            onExit(played.current);
            return;
        }
        if (index > 0)
            beep(true);
        speak(announce(steps[index]));
    // 只在步骤变化时触发；onExit 和 total 在一次跟练中不变
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [index, finished, steps]);

    // 最后 3 秒倒计时提示音
    const lastBeep = useRef(-1);
    useEffect(() => {
        if (finished || paused || remaining > 3 || remaining < 1)
            return;
        const key = index * 10 + remaining;
        if (key === lastBeep.current)
            return;
        lastBeep.current = key;
        beep();
    }, [remaining, index, finished, paused]);

    // 正常做完时不打断"训练完成"的播报
    const finishedRef = useRef(false);
    finishedRef.current = finished;
    useEffect(() => () => {
        if (!finishedRef.current)
            stopSpeaking();
    }, []);

    function jumpTo(i: number) {
        const target = stepStart(steps, Math.min(i, steps.length)) * 1000;
        closeSegment(elapsed);
        segStart.current = target / 1000;
        baseMs.current = target;
        startedAt.current = performance.now();
        setElapsedMs(target);
    }

    function exit() {
        if (!window.confirm('确定结束本次训练吗？'))
            return;
        closeSegment(elapsed);
        onExit(played.current);
    }

    if (finished)
        return <div className="player done" />;

    const ex = EXERCISE_BY_ID[step.exerciseId];
    const block = workout.blocks[step.blockIndex];
    const isWork = step.kind === 'work';
    const repNow = isWork && step.reps !== undefined
        ? Math.min(step.reps, Math.floor(intoStep / ex.secPerRep) + 1)
        : undefined;
    const nextWork = steps.slice(index + 1).find((s) => s.kind === 'work');

    return (
        <div className={`player ${step.kind}`}>
            <div className="player-top">
                <span>{block.title}</span>
                <span>剩余 {formatDuration(Math.ceil(total - elapsed))}</span>
            </div>
            <div className="progress">
                <div style={{ width: `${(elapsed / total) * 100}%` }} />
            </div>

            <div className="player-main">
                <div className="phase-tag">
                    {step.kind === 'prepare' ? '准备' : isWork ? '开始' : step.label === '换另一侧' ? '换边' : '休息'}
                    {step.round && ` · 第 ${step.round.index}/${step.round.count} 轮`}
                    {step.set && ` · 第 ${step.set.index}/${step.set.count} 组`}
                    {step.side && ` · ${step.side === 'left' ? '左侧' : '右侧'}`}
                </div>
                <div className="big-title">{isWork ? ex.name : `下一个：${ex.name}`}</div>
                <div className="countdown">{remaining}</div>
                {repNow !== undefined && (
                    <div className="reps">第 {repNow} / {step.reps} 次</div>
                )}
                {isWork && clipUrls(step.exerciseId, step.side, coach).length > 0 ? (
                    <ExerciseVideo step={step} intoStep={intoStep} paused={paused} coach={coach} />
                ) : hasAnimation(step.exerciseId) && (
                    // 没有视频时用 3D 人偶动画；休息时预演下一个动作
                    <ExerciseAnimation step={step} intoStep={intoStep} paused={paused} coach={coach} />
                )}
                <ul className="cues">
                    {ex.cues.map((c) => (
                        <li key={c}>{c}</li>
                    ))}
                </ul>
                <ExerciseBodyMap exerciseId={ex.id} stretch={ex.category === 'stretch'} />
                {isWork && nextWork && (
                    <div className="muted">之后：{EXERCISE_BY_ID[nextWork.exerciseId].name}</div>
                )}
            </div>

            <div className="row center">
                <button type="button" onClick={exit}>结束</button>
                <button type="button" className="primary" onClick={() => setPaused(!paused)}>
                    {paused ? '继续' : '暂停'}
                </button>
                <button type="button" onClick={() => jumpTo(index + 1)}>跳过</button>
            </div>
        </div>
    );
}
