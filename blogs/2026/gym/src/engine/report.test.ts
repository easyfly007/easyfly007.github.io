import { describe, expect, it } from 'vitest';
import type { Block, WorkoutRecord } from '../types';
import { BASE } from './fixtures';
import { generateWorkout } from './generate';
import { dailyTotals, monthGrid, monthSummary, streakDays, summarize, weeklyTotals } from './report';
import { buildTimeline, PREPARE_SEC, totalSeconds } from './timeline';

// 准备 10 秒 → 深蹲 30 秒（10 次 × 3 秒）→ 休息 30 → 深蹲 30 → 休息 30 → 弓步 左 30 / 换边 5 / 右 30
const BLOCK: Block = {
    phase: 'main',
    title: 't',
    format: 'straight',
    rounds: 1,
    restBetweenRoundsSec: 0,
    items: [
        { exerciseId: 'squat', sets: 2, reps: 10, restSec: 30 },
        { exerciseId: 'lunge', sets: 1, reps: 10, restSec: 30 },
    ],
};

function record(played: [number, number][], startedAt = '2026-10-10T08:00:00'): WorkoutRecord {
    const workout = { ...generateWorkout(BASE, 1), blocks: [BLOCK] };
    return { id: 'r', workout, startedAt, endedAt: startedAt, played };
}

const TOTAL = totalSeconds(buildTimeline([BLOCK]));

describe('summarize', () => {
    it('counts a fully played workout as complete', () => {
        const s = summarize(record([[0, TOTAL]]));
        expect(s.completion).toBe(1);
        expect(s.setsDone).toBe(3);
        expect(s.setsPlanned).toBe(3);
        expect(s.exercises.map((e) => [e.exerciseId, e.reps])).toEqual([['squat', 20], ['lunge', 20]]);
        expect(s.workSec).toBe(120);
    });

    it('does not count skipped parts', () => {
        // 做完第一组深蹲，跳到弓步，弓步左侧做一半就结束
        const lungeStart = PREPARE_SEC + 30 + 30 + 30 + 30;
        const s = summarize(record([[0, PREPARE_SEC + 30], [lungeStart, lungeStart + 15]]));
        expect(s.doneSec).toBe(PREPARE_SEC + 45);
        expect(s.setsDone).toBe(1);
        const lunge = s.exercises.find((e) => e.exerciseId === 'lunge')!;
        expect(lunge.setsDone).toBe(0);
        expect(lunge.reps).toBe(5);
        expect(s.completion).toBeCloseTo((PREPARE_SEC + 45) / TOTAL);
    });

    it('estimates calories only when the weight is known', () => {
        expect(summarize(record([[0, TOTAL]])).kcal).toBeUndefined();
        const light = summarize(record([[0, TOTAL]]), 50).kcal!;
        const heavy = summarize(record([[0, TOTAL]]), 100).kcal!;
        expect(light).toBeGreaterThan(0);
        expect(heavy).toBeCloseTo(light * 2, -1);
    });

    it('lists trained muscles, most trained first', () => {
        const s = summarize(record([[0, TOTAL]]));
        expect(s.muscles.length).toBeGreaterThan(0);
        expect(s.muscles[0].seconds).toBeGreaterThanOrEqual(s.muscles[s.muscles.length - 1].seconds);
    });
});

describe('history aggregation', () => {
    const today = new Date(2026, 9, 10, 20, 0); // 2026-10-10 周六
    const full: [number, number][] = [[0, TOTAL]];
    const records = [
        record(full, new Date(2026, 9, 10, 7).toISOString()),
        record(full, new Date(2026, 9, 9, 7).toISOString()),
        record(full, new Date(2026, 9, 9, 19).toISOString()),
        record(full, new Date(2026, 9, 1, 7).toISOString()),
    ];

    it('sums the last 7 days, oldest first', () => {
        const days = dailyTotals(records, today);
        expect(days).toHaveLength(7);
        expect(days.map((d) => d.count)).toEqual([0, 0, 0, 0, 0, 2, 1]);
        expect(days[6].seconds).toBe(TOTAL);
    });

    it('sums weeks starting on Monday', () => {
        const weeks = weeklyTotals(records, today);
        expect(weeks).toHaveLength(8);
        expect(weeks[7].key).toBe('2026-10-05');
        expect(weeks[7].count).toBe(3);
        expect(weeks[6].count).toBe(1);
    });

    it('counts consecutive training days', () => {
        expect(streakDays(records, today)).toBe(2);
        expect(streakDays(records, new Date(2026, 9, 11))).toBe(2);
        expect(streakDays(records, new Date(2026, 9, 12))).toBe(0);
    });
});

describe('month calendar', () => {
    it('lays out a month in Monday-first weeks', () => {
        const weeks = monthGrid(2026, 9); // 2026 年 10 月，1 日是周四
        expect(weeks[0].slice(0, 3)).toEqual([null, null, null]);
        expect(weeks[0][3]!.getDate()).toBe(1);
        expect(weeks.flat().filter(Boolean)).toHaveLength(31);
        expect(weeks.every((w) => w.length === 7)).toBe(true);
    });

    it('sums one month of training by day', () => {
        const full: [number, number][] = [[0, TOTAL]];
        const records = [
            record(full, new Date(2026, 9, 9, 7).toISOString()),
            record(full, new Date(2026, 9, 9, 19).toISOString()),
            record(full, new Date(2026, 9, 1, 7).toISOString()),
            record(full, new Date(2026, 8, 30, 7).toISOString()),
        ];
        const m = monthSummary(records, 2026, 9, 60);
        expect(m.count).toBe(3);
        expect(m.activeDays).toBe(2);
        expect(m.seconds).toBe(TOTAL * 3);
        expect(m.byDay.get('2026-10-09')!.records).toHaveLength(2);
        expect(m.kcal).toBeGreaterThan(0);
        expect(m.muscles.length).toBeGreaterThan(0);
    });
});
