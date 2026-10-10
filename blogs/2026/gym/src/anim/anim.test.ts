import { describe, expect, it } from 'vitest';
import { EXERCISES, EXERCISE_BY_ID } from '../data/exercises';
import { createFigure } from './figure';
import { ANIMATIONS } from './library';
import { mirrorDef, poseAt } from './pose';
import { animCycleSec, viewIndexAt } from './schedule';

const PHASES = Array.from({ length: 16 }, (_, i) => i / 16);

describe('animation library', () => {
    it('has an animation for every exercise', () => {
        expect(EXERCISES.filter((e) => !(e.id in ANIMATIONS)).map((e) => e.id)).toEqual([]);
    });

    it('only animates known exercises, with keyframes starting at 0 in order', () => {
        for (const [id, def] of Object.entries(ANIMATIONS)) {
            expect(EXERCISE_BY_ID[id], id).toBeDefined();
            expect(def.keys[0][0], id).toBe(0);
            const ts = def.keys.map(([t]) => t);
            expect([...ts].sort((a, b) => a - b), id).toEqual(ts);
            expect(ts[ts.length - 1], id).toBeLessThan(1);
        }
    });

    it('keeps every animated figure on the ground and anchored parts in place', () => {
        for (const [id, base] of Object.entries(ANIMATIONS)) {
            const defs = EXERCISE_BY_ID[id].perSide ? [base, mirrorDef(base)] : [base];
            for (const def of defs) {
                const fig = createFigure('male', def);
                let anchor0: { x: number; z: number } | null = null;
                for (const ph of PHASES) {
                    const p = poseAt(def, ph);
                    fig.setPose(p, ph);
                    const j = fig.joints();
                    const lowest = Math.min(...Object.values(j).map((v) => v.y));
                    // 关节都在地面以上（贴地后最低的身体表面在 y = lift）
                    expect(lowest, `${id} @${ph}`).toBeGreaterThan(-0.01);
                    if (def.anchor === 'feet') {
                        const a = { x: (j.ankleL.x + j.ankleR.x) / 2, z: (j.ankleL.z + j.ankleR.z) / 2 };
                        anchor0 ??= a;
                        expect(Math.hypot(a.x - anchor0.x, a.z - anchor0.z), `${id} @${ph}`).toBeLessThan(0.01);
                    }
                }
            }
        }
    });
});

describe('animation schedule', () => {
    it('changes view after every squat rep', () => {
        expect(animCycleSec('squat')).toBe(3);
        expect([0.5, 3.5, 6.5, 9.5].map((t) => viewIndexAt('squat', t))).toEqual([0, 1, 2, 0]);
    });

    it('groups very short cycles so each view lasts at least 2.5 seconds', () => {
        // 开合跳一次 1.2 秒 → 每 3 次（3.6 秒）换一个视角
        expect(viewIndexAt('jumping_jacks', 3.5)).toBe(0);
        expect(viewIndexAt('jumping_jacks', 3.7)).toBe(1);
    });
});
