import { describe, expect, it } from 'vitest';
import { generateWorkout } from '../engine/generate';
import { BASE } from '../engine/fixtures';
import { buildProfile, mergeProfile, parseProfile, profileText } from './profile';
import { createSavedStore, type KeyValueStore } from './saved';

function memoryStore(): KeyValueStore {
    const m = new Map<string, string>();
    return {
        getItem: (k) => m.get(k) ?? null,
        setItem: (k, v) => {
            m.set(k, v);
        },
    };
}

describe('profile file', () => {
    it('round-trips through the file format and merges without duplicates', () => {
        const a = createSavedStore(memoryStore());
        const w = generateWorkout(BASE, 1);
        a.addRecord({ workout: w, startedAt: '2026-10-08T08:00:00Z', endedAt: '2026-10-08T08:30:00Z', played: [[0, 1800]] });
        a.saveFavorites(['squat']);
        a.setWeightKg(62);
        const text = profileText(buildProfile(a, '小明'));

        const b = createSavedStore(memoryStore());
        b.addRecord({ workout: w, startedAt: '2026-10-09T08:00:00Z', endedAt: '2026-10-09T08:20:00Z', played: [[0, 1200]] });
        b.saveFavorites(['plank']);
        expect(mergeProfile(b, parseProfile(text)).added).toBe(1);
        expect(b.history().map((r) => r.startedAt)).toEqual(['2026-10-09T08:00:00Z', '2026-10-08T08:00:00Z']);
        expect(b.favorites().sort()).toEqual(['plank', 'squat']);
        expect(b.weightKg()).toBe(62);
        expect(b.profileInfo().name).toBe('小明');
        // 再合并一次不会重复
        expect(mergeProfile(b, parseProfile(text)).added).toBe(0);
        expect(b.history()).toHaveLength(2);
    });

    it('rejects files that are not training profiles', () => {
        expect(() => parseProfile('not json')).toThrow('JSON');
        expect(() => parseProfile('{"a":1}')).toThrow('训练档案');
    });
});
