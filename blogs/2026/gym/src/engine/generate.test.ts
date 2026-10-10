import { describe, expect, it } from 'vitest';
import { EXERCISE_BY_ID } from '../data/exercises';
import type { Conditions, Equipment, Goal, Intensity, RestLevel } from '../types';
import { isFeasible, maxSweat } from './filter';
import { BASE } from './fixtures';
import { generateWorkout } from './generate';
import { buildTimeline, locate, stepStart, totalSeconds } from './timeline';

const DURATIONS = [15, 20, 30, 45, 60];
const GOALS: Goal[] = ['general', 'fat_loss', 'strength', 'flexibility'];
const INTENSITIES: Intensity[] = [1, 2, 3];
const RESTS: RestLevel[] = ['short', 'normal', 'long'];
const KITS: Equipment[][] = [[], ['mat'], ['dumbbell', 'mat'], ['band', 'chair'], ['barbell', 'kettlebell', 'mat']];

function allConditions(): Conditions[] {
    const out: Conditions[] = [];
    for (const durationMin of DURATIONS) {
        for (const goal of GOALS) {
            for (const intensity of INTENSITIES) {
                for (const equipment of KITS) {
                    for (const canShower of [true, false]) {
                        for (const rest of RESTS)
                            out.push({ ...BASE, durationMin, goal, intensity, equipment, canShower, rest });
                    }
                }
            }
        }
    }
    return out;
}

describe('generateWorkout', () => {
    it('is deterministic for the same seed', () => {
        expect(generateWorkout(BASE, 42)).toEqual(generateWorkout(BASE, 42));
    });

    it('varies with the seed', () => {
        const ids = (seed: number) =>
            generateWorkout(BASE, seed).blocks.flatMap((b) => b.items.map((i) => i.exerciseId)).join();
        const distinct = new Set([1, 2, 3, 4, 5].map(ids));
        expect(distinct.size).toBeGreaterThan(1);
    });

    it('has warmup, main and cooldown', () => {
        const w = generateWorkout(BASE, 1);
        expect(w.blocks.map((b) => b.phase)).toEqual(['warmup', 'main', 'cooldown']);
    });

    it('lands close to the requested duration and only uses feasible exercises', () => {
        for (const c of allConditions()) {
            for (const seed of [1, 7]) {
                const w = generateWorkout(c, seed);
                const sec = totalSeconds(buildTimeline(w.blocks));
                const tag = JSON.stringify({ ...c, seed, sec });
                expect(Math.abs(sec - c.durationMin * 60) / (c.durationMin * 60), tag).toBeLessThan(0.15);
                expect(w.estimatedMin).toBe(Math.round(sec / 60));
                for (const b of w.blocks) {
                    for (const it of b.items) {
                        const e = EXERCISE_BY_ID[it.exerciseId];
                        expect(isFeasible(e, c), `${e.id} ${tag}`).toBe(true);
                        expect(e.sweat, `${e.id} ${tag}`).toBeLessThanOrEqual(maxSweat(c));
                    }
                }
            }
        }
    });

    it('uses low-sweat straight sets when the user cannot shower', () => {
        const w = generateWorkout({ ...BASE, intensity: 3, canShower: false, goal: 'fat_loss' }, 3);
        expect(w.blocks[1].format).toBe('straight');
    });

    it('uses a circuit for fat loss', () => {
        const w = generateWorkout({ ...BASE, goal: 'fat_loss', intensity: 3 }, 3);
        expect(w.blocks[1].format).toBe('circuit');
    });

    it('shifts main-block rest by 15 seconds per level, never below 10', () => {
        const restOf = (c: Conditions) => {
            const main = generateWorkout(c, 3).blocks[1];
            return [main.items[0].restSec, main.restBetweenRoundsSec];
        };
        const strength = { ...BASE, goal: 'strength' as const, intensity: 2 as const };
        expect(restOf({ ...strength, rest: 'short' })[0]).toBe(60);
        expect(restOf(strength)[0]).toBe(75);
        expect(restOf({ ...strength, rest: 'long' })[0]).toBe(90);

        const circuit = { ...BASE, goal: 'fat_loss' as const, intensity: 1 as const };
        expect(restOf({ ...circuit, rest: 'short' })).toEqual([15, 45]);
        expect(restOf({ ...circuit, rest: 'long' })).toEqual([45, 75]);
        expect(restOf({ ...circuit, intensity: 3, rest: 'short' })[0]).toBe(10);
    });

    it('treats a missing rest level as normal (old saved data)', () => {
        const old: Conditions = { ...BASE };
        delete old.rest;
        expect(generateWorkout(old, 4).blocks).toEqual(generateWorkout(BASE, 4).blocks);
    });

    it('prefers favourite exercises when they are feasible', () => {
        const c = { ...BASE, goal: 'strength' as const };
        let withFav = 0;
        let without = 0;
        for (let seed = 1; seed <= 30; seed++) {
            const ids = (fav: string[]) => generateWorkout(c, seed, fav).blocks.flatMap((b) => b.items.map((i) => i.exerciseId));
            if (ids(['wall_sit']).includes('wall_sit'))
                withFav++;
            if (ids([]).includes('wall_sit'))
                without++;
        }
        expect(withFav).toBeGreaterThan(without);
    });

    it('does not repeat an exercise within the main block', () => {
        const w = generateWorkout({ ...BASE, goal: 'strength', equipment: ['dumbbell'] }, 9);
        const ids = w.blocks[1].items.map((i) => i.exerciseId);
        expect(new Set(ids).size).toBe(ids.length);
    });
});

describe('buildTimeline', () => {
    it('starts with prepare and never ends on a rest', () => {
        const steps = buildTimeline(generateWorkout(BASE, 5).blocks);
        expect(steps[0].kind).toBe('prepare');
        expect(steps[steps.length - 1].kind).toBe('work');
    });

    it('splits per-side exercises into left and right', () => {
        const steps = buildTimeline([
            {
                phase: 'main',
                title: 't',
                format: 'straight',
                rounds: 1,
                restBetweenRoundsSec: 0,
                items: [{ exerciseId: 'lunge', sets: 1, reps: 10, restSec: 30 }],
            },
        ]);
        expect(steps.filter((s) => s.kind === 'work').map((s) => s.side)).toEqual(['left', 'right']);
        expect(steps.find((s) => s.kind === 'work')!.seconds).toBe(30);
    });

    it('locates the current step by elapsed seconds', () => {
        const steps = buildTimeline(generateWorkout(BASE, 5).blocks);
        expect(locate(steps, 0)).toEqual({ index: 0, intoStep: 0 });
        expect(locate(steps, stepStart(steps, 3) + 1)).toEqual({ index: 3, intoStep: 1 });
        expect(locate(steps, totalSeconds(steps)).index).toBe(steps.length);
    });
});
