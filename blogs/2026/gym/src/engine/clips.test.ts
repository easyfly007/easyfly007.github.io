import { describe, expect, it } from 'vitest';
import type { Step } from '../types';
import { clipAt, clipName, TIME_CLIP_SEC } from './clips';

// 深蹲每次 3 秒，20 次 = 60 秒
const SQUATS: Step = { kind: 'work', exerciseId: 'squat', seconds: 60, label: '', blockIndex: 0, reps: 20 };

describe('clipAt', () => {
    it('switches view after every rep: front, side, full, front …', () => {
        const views = Array.from({ length: 20 }, (_, i) => clipAt(SQUATS, i * 3 + 1)!.view);
        expect(views.slice(0, 7)).toEqual(['front', 'side', 'full', 'front', 'side', 'full', 'front']);
        expect(clipAt(SQUATS, 3 * 19 + 1)!.cycle).toBe(19);
    });

    it('reports progress inside the current rep and stays in the last rep at the end', () => {
        expect(clipAt(SQUATS, 4.5)).toEqual({ cycle: 1, view: 'side', progress: 0.5 });
        expect(clipAt(SQUATS, 60)).toEqual({ cycle: 19, view: 'side', progress: 1 });
    });

    it('rotates only through the views that exist', () => {
        expect(clipAt(SQUATS, 4, ['front', 'full'])!.view).toBe('full');
        expect(clipAt(SQUATS, 4, [])).toBeNull();
    });

    it('uses fixed-length clips for timed exercises', () => {
        const plank: Step = { kind: 'work', exerciseId: 'plank', seconds: 30, label: '', blockIndex: 0 };
        expect(clipAt(plank, TIME_CLIP_SEC + 1)!.view).toBe('side');
    });

    it('has no clip for rest steps', () => {
        expect(clipAt({ ...SQUATS, kind: 'rest' }, 1)).toBeNull();
    });
});

describe('clipName', () => {
    it('adds the side for one-sided exercises', () => {
        expect(clipName('front')).toBe('front');
        expect(clipName('side', 'left')).toBe('side-left');
    });
});
