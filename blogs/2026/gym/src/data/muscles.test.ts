import { describe, expect, it } from 'vitest';
import { EXERCISES } from './exercises';
import { EXERCISE_MUSCLES } from './muscles';

describe('exercise muscles', () => {
    it('covers every exercise with at least one primary muscle', () => {
        for (const e of EXERCISES) {
            const mm = EXERCISE_MUSCLES[e.id];
            expect(mm, e.id).toBeDefined();
            expect(mm.primary.length, e.id).toBeGreaterThan(0);
        }
    });

    it('has no unknown ids and no muscle listed as both primary and secondary', () => {
        const ids = new Set(EXERCISES.map((e) => e.id));
        for (const [id, mm] of Object.entries(EXERCISE_MUSCLES)) {
            expect(ids.has(id), id).toBe(true);
            expect(mm.primary.filter((x) => mm.secondary.includes(x)), id).toEqual([]);
            expect(new Set([...mm.primary, ...mm.secondary]).size, id).toBe(mm.primary.length + mm.secondary.length);
        }
    });
});
