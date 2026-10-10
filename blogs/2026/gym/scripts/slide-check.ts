// 检查脚（和撑地的手）在贴地时是否水平滑动。
// 用法（在本目录下）：npx vite-node scripts/slide-check.ts [阈值米，默认 0.03]
import { createFigure } from '../src/anim/figure';
import { ANIMATIONS } from '../src/anim/library';
import { mirrorDef, poseAt } from '../src/anim/pose';
import { EXERCISE_BY_ID } from '../src/data/exercises';
import type { AnimDef } from '../src/anim/types';

const limit = Number(process.argv[2] ?? 0.03);
const N = 48;
const PARTS: [string, string, number][] = [["footL", "ankleL", 0.1], ["footR", "ankleR", 0.1], ["handL", "handL", 0.07], ["handR", "handR", 0.07]];

function slides(def: AnimDef): { part: string; slide: number; always: boolean }[] {
    const fig = createFigure('male', def);
    const out: { part: string; slide: number; always: boolean }[] = [];
    const frames = Array.from({ length: N }, (_, i) => {
        fig.setPose(poseAt(def, i / N), i / N);
        return fig.joints();
    });
    for (const [part, joint, h] of PARTS) {
        const grounded = frames.map((j) => j[joint].y < h);
        let slide = 0;
        // 连续贴地的一段里，相对这段开始位置的最大水平位移
        let start: { x: number; z: number } | null = null;
        frames.forEach((j, i) => {
            if (!grounded[i]) {
                start = null;
                return;
            }
            start ??= { x: j[joint].x, z: j[joint].z };
            slide = Math.max(slide, Math.hypot(j[joint].x - start.x, j[joint].z - start.z));
        });
        if (grounded.some(Boolean))
            out.push({ part, slide, always: grounded.every(Boolean) });
    }
    return out;
}

{
    const bad: string[] = [];
    for (const [id, base] of Object.entries(ANIMATIONS)) {
        const defs: [string, AnimDef][] = EXERCISE_BY_ID[id].perSide ? [['左', base], ['右', mirrorDef(base)]] : [['', base]];
        for (const [side, def] of defs) {
            const s = slides(def).filter((x) => x.slide > limit);
            if (s.length)
                bad.push(`${id} ${EXERCISE_BY_ID[id].name}${side} ` + s.map((x) => `${x.part} ${(x.slide * 100).toFixed(0)}cm${x.always ? '' : '（部分时间贴地）'}`).join('，'));
        }
    }
    console.log(`滑动超过 ${limit * 100}cm：${bad.length} 项`);
    console.log(bad.join('\n'));
}
