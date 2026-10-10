import { describe, expect, it } from 'vitest';
import type { Step } from '../types';
import { buildCoaching, tipsOf } from './coaching';

const squat = (seconds: number, reps: number): Step => ({ kind: 'work', exerciseId: 'squat', seconds, label: '', blockIndex: 0, reps });

describe('coaching', () => {
    it('splits cues into short tips', () => {
        expect(tipsOf('squat')).toEqual(['双脚与肩同宽', '脚尖略外八', '臀部向后坐', '膝盖与脚尖同向']);
    });

    it('alternates tips and progress, every 5 seconds, quiet in the last 4 seconds', () => {
        const [ev] = buildCoaching([squat(45, 15)]);
        expect(ev.map((e) => [e.at, e.text])).toEqual([
            [5, '双脚与肩同宽'],
            [10, '还剩 12 次'],
            [15, '脚尖略外八'],
            [20, '还剩 9 次'],
            [25, '臀部向后坐'],
            [30, '还剩 5 次'],
            [35, '膝盖与脚尖同向'],
            [40, '还剩 2 次'],
        ]);
    });

    it('does not repeat tips in later sets of the same exercise', () => {
        const rest: Step = { kind: 'rest', exerciseId: 'squat', seconds: 30, label: '', blockIndex: 0 };
        const [, , second] = buildCoaching([squat(45, 15), rest, squat(45, 15)]);
        expect(second.every((e) => e.kind === 'progress')).toBe(true);
        expect(second.length).toBeGreaterThan(0);
    });
});
