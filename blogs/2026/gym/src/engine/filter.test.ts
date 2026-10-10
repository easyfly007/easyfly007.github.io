import { describe, expect, it } from 'vitest';
import { EXERCISES } from '../data/exercises';
import type { Conditions } from '../types';
import { candidates, isFeasible } from './filter';

export const BASE: Conditions = {
    equipment: [],
    location: 'indoor',
    space: 'small',
    quiet: false,
    hasShoes: true,
    durationMin: 30,
    intensity: 2,
    canShower: true,
    goal: 'general',
    limitations: [],
};

describe('exercise library', () => {
    it('has unique ids', () => {
        const ids = EXERCISES.map((e) => e.id);
        expect(new Set(ids).size).toBe(ids.length);
    });

    it('has a duration source for every exercise', () => {
        for (const e of EXERCISES) {
            if (e.mode === 'reps')
                expect(e.defaultReps, e.id).toBeGreaterThan(0);
            else
                expect(e.defaultSeconds, e.id).toBeGreaterThan(0);
            expect(e.secPerRep, e.id).toBeGreaterThan(0);
        }
    });

    it('keeps sweat and intensity within 1..5', () => {
        for (const e of EXERCISES) {
            expect(e.sweat, e.id).toBeGreaterThanOrEqual(1);
            expect(e.sweat, e.id).toBeLessThanOrEqual(5);
            expect(e.intensity, e.id).toBeGreaterThanOrEqual(1);
            expect(e.intensity, e.id).toBeLessThanOrEqual(5);
        }
    });
});

describe('filter', () => {
    it('excludes exercises needing missing equipment', () => {
        const list = candidates(EXERCISES, BASE, 'main');
        expect(list.every((e) => e.equipment.length === 0)).toBe(true);
    });

    it('includes dumbbell exercises when dumbbells are available', () => {
        const list = candidates(EXERCISES, { ...BASE, equipment: ['dumbbell'] }, 'main');
        expect(list.some((e) => e.equipment.includes('dumbbell'))).toBe(true);
    });

    it('caps sweat at 2 when the user cannot shower', () => {
        const list = candidates(EXERCISES, { ...BASE, intensity: 3, canShower: false }, 'main');
        expect(list.length).toBeGreaterThan(0);
        expect(list.every((e) => e.sweat <= 2)).toBe(true);
    });

    it('drops noisy and shoe-dependent exercises', () => {
        const c: Conditions = { ...BASE, quiet: true, hasShoes: false, intensity: 3 };
        const list = candidates(EXERCISES, c, 'main');
        expect(list.some((e) => e.noisy || e.needsShoes)).toBe(false);
    });

    it('respects body limitations', () => {
        const c: Conditions = { ...BASE, limitations: ['knee'] };
        expect(candidates(EXERCISES, c, 'main').some((e) => e.stresses.includes('knee'))).toBe(false);
    });

    it('treats outdoor-only exercises as infeasible indoors', () => {
        const sprint = EXERCISES.find((e) => e.id === 'sprint')!;
        expect(isFeasible(sprint, BASE)).toBe(false);
        expect(isFeasible(sprint, { ...BASE, location: 'outdoor', space: 'large' })).toBe(true);
    });

    // 最苛刻的条件（徒手、无垫、无鞋、安静、不能洗澡、四处受限）也要有足够动作
    it('leaves enough exercises under the most restrictive conditions', () => {
        const c: Conditions = {
            ...BASE,
            hasShoes: false,
            quiet: true,
            canShower: false,
            intensity: 1,
            limitations: ['knee', 'back', 'shoulder', 'wrist'],
        };
        expect(candidates(EXERCISES, c, 'warmup').length).toBeGreaterThanOrEqual(3);
        expect(candidates(EXERCISES, c, 'main').length).toBeGreaterThanOrEqual(5);
        expect(candidates(EXERCISES, c, 'cooldown').length).toBeGreaterThanOrEqual(3);
    });
});
