import { describe, expect, it } from 'vitest';
import { generateWorkout } from '../engine/generate';
import { BASE } from '../engine/fixtures';
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

describe('saved store', () => {
    it('saves, lists newest first, and removes', () => {
        const s = createSavedStore(memoryStore());
        const a = s.save('A', generateWorkout(BASE, 1));
        const b = s.save('B', generateWorkout(BASE, 2));
        expect(s.list().map((x) => x.name)).toEqual(['B', 'A']);
        s.remove(b.id);
        expect(s.list().map((x) => x.id)).toEqual([a.id]);
    });

    it('round-trips the workout exactly', () => {
        const s = createSavedStore(memoryStore());
        const w = generateWorkout(BASE, 3);
        s.save('W', w);
        expect(s.list()[0].workout).toEqual(w);
    });

    it('renames and marks as used', () => {
        const s = createSavedStore(memoryStore());
        const a = s.save('A', generateWorkout(BASE, 1));
        s.rename(a.id, '办公室-徒手');
        s.markUsed(a.id);
        const [got] = s.list();
        expect(got.name).toBe('办公室-徒手');
        expect(got.lastUsedAt).toBeDefined();
    });

    it('remembers the last conditions', () => {
        const s = createSavedStore(memoryStore());
        expect(s.lastConditions()).toBeNull();
        s.rememberConditions(BASE);
        expect(s.lastConditions()).toEqual(BASE);
    });

    it('survives corrupted data', () => {
        const kv = memoryStore();
        kv.setItem('gym.saved.v1', '{not json');
        expect(createSavedStore(kv).list()).toEqual([]);
    });
});
