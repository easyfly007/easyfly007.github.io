import { EXERCISE_BY_ID } from '../data/exercises';
import type { Muscle, Step, WorkoutRecord } from '../types';
import { buildTimeline, totalSeconds } from './timeline';

// 少于这个秒数的训练（比如点开就结束）不记录
export const MIN_RECORD_SEC = 60;
// 一步做到这个比例就算完成（容忍最后零点几秒的误差）
const STEP_DONE_RATIO = 0.9;
// 按动作强度等级 1..5 的代谢当量（MET）；休息按 1.5
const MET_BY_INTENSITY = [0, 3, 4, 5.5, 7, 8.5];
const REST_MET = 1.5;

export interface ExerciseSummary {
    exerciseId: string;
    name: string;
    setsDone: number;
    setsPlanned: number;
    reps?: number;              // 计次动作实际完成的次数
    seconds: number;            // 实际做的秒数
}

export interface RecordSummary {
    plannedSec: number;
    doneSec: number;
    workSec: number;
    completion: number;         // 0..1
    setsDone: number;
    setsPlanned: number;
    exercises: ExerciseSummary[];
    muscles: { muscle: Muscle; seconds: number }[];
    kcal?: number;              // 估算值，没有体重时不算
}

function overlap(a: [number, number], b: [number, number]): number {
    return Math.max(0, Math.min(a[1], b[1]) - Math.max(a[0], b[0]));
}

// 每一步实际播放了多少秒
export function stepDoneSeconds(steps: Step[], played: [number, number][]): number[] {
    let start = 0;
    return steps.map((s) => {
        const range: [number, number] = [start, start + s.seconds];
        start += s.seconds;
        const done = played.reduce((sum, p) => sum + overlap(p, range), 0);
        return Math.min(s.seconds, done);
    });
}

// 一组 = 一个 work 步骤；单侧动作的左右两步算同一组
function setKey(steps: Step[], i: number): string {
    const s = steps[i];
    if (s.side === 'right') {
        for (let j = i - 1; j >= 0; j--) {
            if (steps[j].kind === 'work' && steps[j].side === 'left')
                return `${j}`;
        }
    }
    return `${i}`;
}

export function playedSeconds(record: WorkoutRecord): number {
    const steps = buildTimeline(record.workout.blocks);
    return stepDoneSeconds(steps, record.played).reduce((a, b) => a + b, 0);
}

export function summarize(record: WorkoutRecord, weightKg?: number): RecordSummary {
    const steps = buildTimeline(record.workout.blocks);
    const done = stepDoneSeconds(steps, record.played);
    const plannedSec = totalSeconds(steps);

    const byExercise = new Map<string, ExerciseSummary>();
    const sets = new Map<string, { exerciseId: string; done: boolean }>();
    const muscles = new Map<Muscle, number>();
    let workSec = 0;
    let metMinutes = 0;

    steps.forEach((s, i) => {
        const sec = done[i];
        if (s.kind !== 'work') {
            if (s.kind === 'rest')
                metMinutes += (REST_MET * sec) / 60;
            return;
        }
        const ex = EXERCISE_BY_ID[s.exerciseId];
        workSec += sec;
        metMinutes += (MET_BY_INTENSITY[ex.intensity] * sec) / 60;
        for (const m of ex.muscles)
            muscles.set(m, (muscles.get(m) ?? 0) + sec);

        let e = byExercise.get(ex.id);
        if (!e) {
            e = { exerciseId: ex.id, name: ex.name, setsDone: 0, setsPlanned: 0, seconds: 0 };
            byExercise.set(ex.id, e);
        }
        e.seconds += sec;
        if (s.reps !== undefined) {
            const reps = Math.min(s.reps, Math.floor(sec / ex.secPerRep + 0.01));
            e.reps = (e.reps ?? 0) + reps;
        }

        const key = setKey(steps, i);
        const stepDone = sec >= s.seconds * STEP_DONE_RATIO;
        const prev = sets.get(key);
        if (prev)
            prev.done = prev.done && stepDone;
        else
            sets.set(key, { exerciseId: ex.id, done: stepDone });
    });

    for (const { exerciseId, done: d } of sets.values()) {
        const e = byExercise.get(exerciseId)!;
        e.setsPlanned += 1;
        if (d)
            e.setsDone += 1;
    }

    const exercises = [...byExercise.values()];
    const doneSec = done.reduce((a, b) => a + b, 0);
    const summary: RecordSummary = {
        plannedSec,
        doneSec,
        workSec,
        completion: plannedSec > 0 ? doneSec / plannedSec : 0,
        setsDone: exercises.reduce((n, e) => n + e.setsDone, 0),
        setsPlanned: exercises.reduce((n, e) => n + e.setsPlanned, 0),
        exercises,
        muscles: [...muscles.entries()]
            .map(([muscle, seconds]) => ({ muscle, seconds }))
            .sort((a, b) => b.seconds - a.seconds),
    };
    // 千卡 = MET × 3.5 × 体重(kg) / 200 × 分钟
    if (weightKg !== undefined && weightKg > 0)
        summary.kcal = Math.round((metMinutes * 3.5 * weightKg) / 200);
    return summary;
}

export interface DayTotal {
    key: string;                // 本地日期 YYYY-MM-DD，周统计时为该周周一
    date: Date;
    seconds: number;
    count: number;
}

export function dayKey(d: Date): string {
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${d.getFullYear()}-${m}-${day}`;
}

function startOfDay(d: Date): Date {
    return new Date(d.getFullYear(), d.getMonth(), d.getDate());
}

function addDays(d: Date, n: number): Date {
    return new Date(d.getFullYear(), d.getMonth(), d.getDate() + n);
}

function mondayOf(d: Date): Date {
    const day = startOfDay(d);
    return addDays(day, -((day.getDay() + 6) % 7));
}

function bucket(records: WorkoutRecord[], starts: Date[], keyOf: (d: Date) => string): DayTotal[] {
    const out = starts.map((date) => ({ key: dayKey(date), date, seconds: 0, count: 0 }));
    const index = new Map(out.map((t, i) => [t.key, i]));
    for (const r of records) {
        const i = index.get(keyOf(new Date(r.startedAt)));
        if (i === undefined)
            continue;
        out[i].seconds += playedSeconds(r);
        out[i].count += 1;
    }
    return out;
}

// 截至 today 的最近 days 天，每天的训练时长与次数（旧 → 新）
export function dailyTotals(records: WorkoutRecord[], today: Date, days = 7): DayTotal[] {
    const end = startOfDay(today);
    const starts = Array.from({ length: days }, (_, i) => addDays(end, i - days + 1));
    return bucket(records, starts, (d) => dayKey(d));
}

// 截至 today 所在周的最近 weeks 周（周一开始），每周的训练时长与次数（旧 → 新）
export function weeklyTotals(records: WorkoutRecord[], today: Date, weeks = 8): DayTotal[] {
    const end = mondayOf(today);
    const starts = Array.from({ length: weeks }, (_, i) => addDays(end, (i - weeks + 1) * 7));
    return bucket(records, starts, (d) => dayKey(mondayOf(d)));
}

// 截至 today 连续有训练的天数；今天还没练则从昨天往前数
export function streakDays(records: WorkoutRecord[], today: Date): number {
    const days = new Set(records.map((r) => dayKey(new Date(r.startedAt))));
    let d = startOfDay(today);
    if (!days.has(dayKey(d)))
        d = addDays(d, -1);
    let n = 0;
    while (days.has(dayKey(d))) {
        n += 1;
        d = addDays(d, -1);
    }
    return n;
}
