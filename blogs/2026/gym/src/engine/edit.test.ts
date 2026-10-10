import { describe, expect, it } from 'vitest';
import { addExercise, removeItem } from './edit';
import { BASE } from './fixtures';
import { generateWorkout } from './generate';
import { buildTimeline, totalSeconds } from './timeline';

function last<T>(list: T[]): T {
    return list[list.length - 1];
}

const minutes = (w: ReturnType<typeof generateWorkout>) => Math.round(totalSeconds(buildTimeline(w.blocks)) / 60);

describe('addExercise', () => {
    it('adds a main exercise with the same sets and rest as the strength block', () => {
        const w = generateWorkout({ ...BASE, goal: 'strength' }, 2);
        const main = w.blocks.find((b) => b.phase === 'main')!;
        const next = addExercise(w, 'glute_bridge');
        const added = last(next.blocks.find((b) => b.phase === 'main')!.items);
        expect(added).toEqual({ exerciseId: 'glute_bridge', sets: main.items[0].sets, reps: 15, restSec: main.items[0].restSec });
        expect(next.estimatedMin).toBe(minutes(next));
        expect(next.estimatedMin).toBeGreaterThan(w.estimatedMin);
    });

    it('uses the interval timing inside a circuit', () => {
        const w = generateWorkout({ ...BASE, goal: 'fat_loss' }, 2);
        const main = w.blocks.find((b) => b.phase === 'main')!;
        expect(main.format).toBe('circuit');
        const added = last(addExercise(w, 'squat').blocks.find((b) => b.phase === 'main')!.items);
        expect(added).toEqual({ exerciseId: 'squat', sets: 1, seconds: main.items[0].seconds, restSec: main.items[0].restSec });
    });

    it('puts warm-up-only and cool-down-only exercises in their own blocks', () => {
        const w = generateWorkout(BASE, 3);
        const a = addExercise(w, 'arm_circles');
        expect(last(a.blocks[0].items).exerciseId).toBe('arm_circles');
        const b = addExercise(w, 'calf_stretch');
        expect(last(last(b.blocks).items).exerciseId).toBe('calf_stretch');
    });
});

describe('removeItem', () => {
    it('removes an item, drops empty blocks and updates the estimate', () => {
        const w = generateWorkout(BASE, 4);
        const first = w.blocks[0];
        let next = w;
        for (let i = 0; i < first.items.length; i++)
            next = removeItem(next, 0, 0);
        expect(next.blocks.length).toBe(w.blocks.length - 1);
        expect(next.estimatedMin).toBe(minutes(next));
    });
});
