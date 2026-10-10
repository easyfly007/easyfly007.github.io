// 动作动画检查图：把每个动作在一个循环里的几个时刻投影成正面、侧面的骨架简图（SVG），
// 用来在没有浏览器时检查姿态和贴地是否正确。左侧肢体橙色，右侧蓝色。
// 用法（在本目录下）：npx vite-node scripts/anim-preview.ts <输出目录> [动作 id ...]
import { mkdirSync, writeFileSync } from 'node:fs';
import { createFigure } from '../src/anim/figure';
import { ANIMATIONS } from '../src/anim/library';
import { poseAt } from '../src/anim/pose';
import { EXERCISE_BY_ID } from '../src/data/exercises';

const outDir = process.argv[2] ?? 'anim-preview';
const only = process.argv.slice(3);
mkdirSync(outDir, { recursive: true });

const PHASES = [0, 0.25, 0.5, 0.75];
const SCALE = 70;               // 像素 / 米
const W = 170;
const H = 190;

const BONES: [string, string, string][] = [
    ['pelvis', 'neck', '#555'],
    ['shoulderL', 'shoulderR', '#555'],
    ['hipL', 'hipR', '#555'],
    ['shoulderL', 'elbowL', '#e0531f'], ['elbowL', 'handL', '#e0531f'],
    ['shoulderR', 'elbowR', '#3b6fd8'], ['elbowR', 'handR', '#3b6fd8'],
    ['hipL', 'kneeL', '#e0531f'], ['kneeL', 'ankleL', '#e0531f'], ['ankleL', 'toeL', '#e0531f'],
    ['hipR', 'kneeR', '#3b6fd8'], ['kneeR', 'ankleR', '#3b6fd8'], ['ankleR', 'toeR', '#3b6fd8'],
];

type P = { x: number; y: number; z: number };

function panel(j: Record<string, P>, view: 'front' | 'side', ox: number, oy: number, label: string, cx: number, cz: number): string {
    // 正面：从人物身后看（人物的左侧在画面左边）；侧面：从人物左侧看（人物前方在画面右边）。画面以身体中心为准
    const px = (p: { x: number; z: number }) => ox + W / 2 + (view === 'front' ? -(p.x - cx) : p.z - cz) * SCALE;
    const py = (p: { y: number }) => oy + H - 20 - p.y * SCALE;
    let s = `<rect x="${ox}" y="${oy}" width="${W}" height="${H}" fill="#fff" stroke="#ddd"/>`;
    s += `<line x1="${ox}" x2="${ox + W}" y1="${py({ y: 0 })}" y2="${py({ y: 0 })}" stroke="#999"/>`;
    for (const [a, b, c] of BONES)
        s += `<line x1="${px(j[a])}" y1="${py(j[a])}" x2="${px(j[b])}" y2="${py(j[b])}" stroke="${c}" stroke-width="4" stroke-linecap="round"/>`;
    s += `<circle cx="${px(j.head)}" cy="${py(j.head)}" r="${0.11 * SCALE}" fill="none" stroke="#555" stroke-width="3"/>`;
    s += `<text x="${ox + 4}" y="${oy + 12}" font-size="10" fill="#333">${label}</text>`;
    return s;
}

const ids = only.length ? only : Object.keys(ANIMATIONS);
for (const id of ids) {
    const def = ANIMATIONS[id];
    const fig = createFigure('male', def);
    let body = '';
    PHASES.forEach((ph, i) => {
        fig.setPose(poseAt(def, ph), ph);
        const j = fig.joints();
        const pts = Object.values(j);
        const cx = pts.reduce((a, p) => a + p.x, 0) / pts.length;
        const cz = pts.reduce((a, p) => a + p.z, 0) / pts.length;
        body += panel(j, 'front', i * W, 0, `${ph} front`, cx, cz);
        body += panel(j, 'side', i * W, H, `${ph} side`, cx, cz);
    });
    const name = EXERCISE_BY_ID[id]?.name ?? id;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W * PHASES.length}" height="${H * 2 + 16}">`
        + `<rect width="100%" height="100%" fill="#fff"/>${body}`
        + `<text x="4" y="${H * 2 + 12}" font-size="11">${id} ${name}</text></svg>`;
    writeFileSync(`${outDir}/${id}.svg`, svg);
}
console.log(`${ids.length} 个动作 → ${outDir}/`);
