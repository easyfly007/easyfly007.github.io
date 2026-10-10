import type { AnimDef, Pose, V3 } from './types';

const RAD = Math.PI / 180;

function norm(v: V3): V3 {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
}

// 肢体方向：从"自然下垂"开始，向前抬 f 度（90 = 水平向前，180 = 向上），向外侧抬 a 度（90 = 水平向外）。
// side：左 1，右 -1
export function limb(f: number, a = 0, side = 1): V3 {
    const fr = f * RAD;
    const ar = a * RAD;
    return norm([side * Math.sin(ar), -Math.cos(fr) * Math.cos(ar), Math.sin(fr) * Math.cos(ar)]);
}

// 躯干/头顶方向：从竖直向上开始，向前倾 f 度，向左侧弯 s 度
export function up(f: number, s = 0): V3 {
    const fr = f * RAD;
    const sr = s * RAD;
    return norm([Math.sin(sr), Math.cos(fr) * Math.cos(sr), Math.sin(fr) * Math.cos(sr)]);
}

// 腿：大腿前抬 f 度、外展 a 度，膝盖弯 k 度（小腿相对大腿向后折）
export function leg(f: number, k = 0, a = 0, side = 1): { thigh: V3; shin: V3 } {
    return { thigh: limb(f, a, side), shin: limb(f - k, a, side) };
}

// 手臂：上臂前抬 f 度、外展 a 度，手肘弯 e 度（前臂相对上臂向前上方折）
export function arm(f: number, a = 0, e = 0, side = 1): { arm: V3; fore: V3 } {
    return { arm: limb(f, a, side), fore: limb(f + e, a, side) };
}

// 两侧对称的腿 / 手臂，直接展开成姿态字段
export function legs(f: number, k = 0, a = 0): Pose {
    const l = leg(f, k, a, 1);
    const r = leg(f, k, a, -1);
    return { thighL: l.thigh, shinL: l.shin, thighR: r.thigh, shinR: r.shin };
}

export function legL(f: number, k = 0, a = 0): Pose {
    const l = leg(f, k, a, 1);
    return { thighL: l.thigh, shinL: l.shin };
}

export function legR(f: number, k = 0, a = 0): Pose {
    const r = leg(f, k, a, -1);
    return { thighR: r.thigh, shinR: r.shin };
}

export function arms(f: number, a = 0, e = 0): Pose {
    const l = arm(f, a, e, 1);
    const r = arm(f, a, e, -1);
    return { armL: l.arm, foreL: l.fore, armR: r.arm, foreR: r.fore };
}

export function armL(f: number, a = 0, e = 0): Pose {
    const l = arm(f, a, e, 1);
    return { armL: l.arm, foreL: l.fore };
}

export function armR(f: number, a = 0, e = 0): Pose {
    const r = arm(f, a, e, -1);
    return { armR: r.arm, foreR: r.fore };
}

// 站立默认姿态
export const STAND: Required<Pick<Pose, 'pitch' | 'roll' | 'yaw' | 'lift' | 'twist' | 'spine' | 'head'>> & Pose = {
    pitch: 0,
    roll: 0,
    yaw: 0,
    lift: 0,
    twist: 0,
    spine: [0, 1, 0],
    head: [0, 1, 0],
    ...arms(0, 8),
    ...legs(0),
};

const VECTOR_KEYS = ['spine', 'head', 'armL', 'foreL', 'armR', 'foreR', 'thighL', 'shinL', 'thighR', 'shinR', 'toeL', 'toeR'] as const;
const NUMBER_KEYS = ['pitch', 'roll', 'yaw', 'lift', 'twist', 'dx', 'dz'] as const;

export function fullPose(p: Pose): Pose {
    return { ...STAND, ...p };
}

function lerpV(a: V3, b: V3, t: number): V3 {
    return norm([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
}

function mix(a: Pose, b: Pose, t: number): Pose {
    const out: Pose = { armsWorld: t < 0.5 ? a.armsWorld : b.armsWorld, legsWorld: t < 0.5 ? a.legsWorld : b.legsWorld };
    for (const k of NUMBER_KEYS)
        out[k] = (a[k] ?? 0) + ((b[k] ?? 0) - (a[k] ?? 0)) * t;
    for (const k of VECTOR_KEYS) {
        const va = a[k];
        const vb = b[k];
        if (va && vb)
            out[k] = lerpV(va, vb, t);
        else
            out[k] = t < 0.5 ? va : vb;
    }
    return out;
}

// 循环内进度 phase（0..1）处的姿态：关键帧之间插值，最后一帧接回第一帧
export function poseAt(def: AnimDef, phase: number): Pose {
    const keys = def.keys;
    const p = ((phase % 1) + 1) % 1;
    for (let i = 0; i < keys.length; i++) {
        const [pa, a] = keys[i];
        const [pb, b] = i + 1 < keys.length ? keys[i + 1] : [1, keys[0][1]];
        if (p >= pa && p <= pb) {
            const u = pb > pa ? (p - pa) / (pb - pa) : 0;
            const t = def.linear ? u : (1 - Math.cos(Math.PI * u)) / 2;
            return mix(fullPose(a), fullPose(b), t);
        }
    }
    return fullPose(keys[0][1]);
}

const SWAP: [keyof Pose, keyof Pose][] = [['armL', 'armR'], ['foreL', 'foreR'], ['thighL', 'thighR'], ['shinL', 'shinR'], ['toeL', 'toeR']];

function flipX(v: V3 | undefined): V3 | undefined {
    return v ? [-v[0], v[1], v[2]] : undefined;
}

// 左右镜像：单侧动作只写左侧，右侧由此生成
export function mirrorPose(p: Pose): Pose {
    const out: Pose = { ...p };
    for (const [l, r] of SWAP) {
        (out as Record<string, unknown>)[l] = flipX(p[r] as V3 | undefined);
        (out as Record<string, unknown>)[r] = flipX(p[l] as V3 | undefined);
    }
    out.spine = flipX(p.spine);
    out.head = flipX(p.head);
    out.roll = p.roll === undefined ? undefined : -p.roll;
    out.yaw = p.yaw === undefined ? undefined : -p.yaw;
    out.twist = p.twist === undefined ? undefined : -p.twist;
    out.dx = p.dx === undefined ? undefined : -p.dx;
    return out;
}

export function mirrorDef(def: AnimDef): AnimDef {
    const swapAnchor = { footL: 'footR', footR: 'footL' } as const;
    const anchor = def.anchor === 'footL' || def.anchor === 'footR' ? swapAnchor[def.anchor] : def.anchor;
    return {
        ...def,
        anchor,
        keys: def.keys.map(([t, p]) => [t, mirrorPose(fullPose(p))]),
        props: def.props?.map((x) => (x === 'dumbbellL' ? 'dumbbellR' : x === 'dumbbellR' ? 'dumbbellL' : x)),
        scene: def.scene?.map((s) => ({
            ...s,
            at: s.at === 'footL' ? 'footR' : s.at === 'footR' ? 'footL' : s.at,
            offset: s.offset ? [-s.offset[0], s.offset[1], s.offset[2]] : undefined,
        })),
    };
}
