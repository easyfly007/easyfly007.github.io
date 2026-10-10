import { armL, armR, arms, legL, legR, legs, limb, up } from './pose';
import type { AnimDef, Pose, V3 } from './types';

// 每个动作的人偶动画：一个循环（计次动作 = 一次动作）的关键帧。
// 单侧动作只写左侧，右侧自动镜像。首尾姿态相同，循环播放时动作连贯

// ---------- 常用姿态片段 ----------

const STANDING: Pose = { ...legs(0, 0, 5), ...arms(0, 8) };

// 深蹲到底：大腿接近水平，小腿前倾，躯干前倾
const squatDown = (lean = 40): Pose => ({ ...legs(95, 125, 12), spine: up(lean) });

// 胸前托举（高脚杯）：手在下巴前
const GOBLET_ARMS: Pose = { ...arms(25, 18, 125) };

// 杠铃扛在背上时的手臂
const BACK_BAR_ARMS: Pose = {
    armL: limb(-30, 55, 1),
    foreL: [-0.05, 1, -0.15],
    armR: limb(-30, 55, -1),
    foreR: [0.05, 1, -0.15],
};

// 仰卧屈膝（双脚踩地）：世界坐标，人物头朝 -z、脚朝 +z
const SUPINE_KNEES_BENT: Pose = {
    pitch: -90,
    legsWorld: true,
    thighL: [0.03, 0.73, 0.68],
    shinL: [0, -0.93, -0.36],
    thighR: [-0.03, 0.73, 0.68],
    shinR: [0, -0.93, -0.36],
    toeL: [0, 0, 1],
    toeR: [0, 0, 1],
};

const straightArmsDown = (x = 0.14, z = 0.04): Pose => ({
    armsWorld: true,
    armL: [x, -1, z],
    foreL: [x, -1, z],
    armR: [-x, -1, z],
    foreR: [-x, -1, z],
});

function squatDef(down: Pose, upPose: Pose = {}, extra: Partial<AnimDef> = {}): AnimDef {
    return {
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, ...upPose }],
            [0.5, { ...STANDING, ...upPose, ...down }],
        ],
        ...extra,
    };
}

const v = (x: number, y: number, z: number): V3 => [x, y, z];

export const ANIMATIONS: Record<string, AnimDef> = {
    // ---------- 深蹲类 ----------
    squat: squatDef({ ...squatDown(), armsWorld: true, ...arms(90, 8) }, { armsWorld: true, ...arms(0, 8) }),

    tempo_squat: {
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, armsWorld: true, ...arms(0, 8) }],
            [0.6, { ...squatDown(), armsWorld: true, ...arms(90, 8) }],
            [0.8, { ...squatDown(), armsWorld: true, ...arms(90, 8) }],
        ],
    },

    db_goblet_squat: squatDef(squatDown(25), GOBLET_ARMS, { props: ['dumbbellGoblet'] }),

    kb_goblet_squat: squatDef(squatDown(25), GOBLET_ARMS, { props: ['kettlebellGoblet'] }),

    bb_squat: squatDef(squatDown(35), BACK_BAR_ARMS, { props: ['barbellBack'] }),

    wall_sit: {
        cycleSec: 4,
        anchor: 'feet',
        scene: [{ kind: 'wall', at: 'back', offset: [0, 0, -0.03] }],
        keys: [
            [0, { ...legs(90, 90, 6), ...arms(35, 12, 45) }],
            [0.5, { ...legs(90, 90, 6), ...arms(35, 12, 45), spine: up(2) }],
        ],
    },

    calf_raise: {
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, toeL: [0, 0, 1], toeR: [0, 0, 1] }],
            [0.5, { ...STANDING, toeL: [0, -0.65, 0.76], toeR: [0, -0.65, 0.76] }],
        ],
    },

    // ---------- 弓步类 ----------
    lunge: {
        anchor: 'footR',
        keys: [
            [0, STANDING],
            [0.5, { ...legL(80, 85, 4), ...legR(-15, 85, 4), spine: up(4) }],
        ],
    },

    // ---------- 臀桥 / 卷腹 ----------
    glute_bridge: {
        anchor: 'feet',
        keys: [
            [0, { ...SUPINE_KNEES_BENT, head: up(30), armsWorld: true, armL: v(0.15, -0.05, 1), foreL: v(0.15, -0.05, 1), armR: v(-0.15, -0.05, 1), foreR: v(-0.15, -0.05, 1) }],
            [0.5, {
                ...SUPINE_KNEES_BENT,
                pitch: -122,
                thighL: [0.03, 0.18, 0.98],
                thighR: [-0.03, 0.18, 0.98],
                shinL: [0, -0.97, -0.24],
                shinR: [0, -0.97, -0.24],
                head: up(55),
                armsWorld: true,
                armL: v(0.15, -0.05, 1),
                foreL: v(0.15, -0.05, 1),
                armR: v(-0.15, -0.05, 1),
                foreR: v(-0.15, -0.05, 1),
            }],
        ],
    },

    crunch: {
        anchor: 'feet',
        keys: [
            [0, { ...SUPINE_KNEES_BENT, ...arms(15, 10) }],
            [0.5, { ...SUPINE_KNEES_BENT, ...arms(30, 10), spine: up(35), head: up(15) }],
        ],
    },

    // ---------- 俯卧撑 / 平板 ----------
    pushup: {
        anchor: 'feet',
        keys: [
            [0, { pitch: 64, ...straightArmsDown(0.18, 0.05) }],
            [0.5, { pitch: 80, armsWorld: true, armL: v(0.55, -0.25, -0.8), foreL: v(0.05, -1, 0.08), armR: v(-0.55, -0.25, -0.8), foreR: v(-0.05, -1, 0.08) }],
        ],
    },

    knee_pushup: {
        anchor: 'feet',
        keys: [
            [0, { pitch: 46, ...legs(0, 100), ...straightArmsDown(0.18, 0.05) }],
            [0.5, { pitch: 73, ...legs(0, 100), armsWorld: true, armL: v(0.55, -0.25, -0.8), foreL: v(0.05, -1, 0.08), armR: v(-0.55, -0.25, -0.8), foreR: v(-0.05, -1, 0.08) }],
        ],
    },

    plank: {
        cycleSec: 4,
        keys: [
            [0, { pitch: 76, armsWorld: true, armL: v(0.05, -1, 0), foreL: v(0.08, -0.05, 1), armR: v(-0.05, -1, 0), foreR: v(-0.08, -0.05, 1) }],
            [0.5, { pitch: 77, armsWorld: true, armL: v(0.05, -1, 0), foreL: v(0.08, -0.05, 1), armR: v(-0.05, -1, 0), foreR: v(-0.08, -0.05, 1), spine: up(2) }],
        ],
    },

    mountain_climber: {
        cycleSec: 0.8,
        anchor: 'hands',
        keys: [
            [0, { pitch: 64, ...straightArmsDown(0.16, 0.03), ...legL(70, 100), ...legR(0) }],
            [0.5, { pitch: 64, ...straightArmsDown(0.16, 0.03), ...legL(0), ...legR(70, 100) }],
        ],
    },

    // ---------- 有氧 ----------
    jumping_jacks: {
        cycleSec: 1.2,
        anchor: 'feet',
        keys: [
            [0, { ...legs(0, 5, 3), ...arms(0, 8) }],
            [0.25, { ...legs(0, 5, 9), ...arms(0, 95), lift: 0.06 }],
            [0.5, { ...legs(0, 12, 16), ...arms(0, 170) }],
            [0.75, { ...legs(0, 5, 9), ...arms(0, 95), lift: 0.06 }],
        ],
    },

    high_knees: {
        cycleSec: 0.8,
        keys: [
            [0, { ...legL(90, 95, 4), ...legR(0, 5, 4), ...armR(60, 8, 70), ...armL(-30, 8, 80), lift: 0.03 }],
            [0.25, { ...legL(25, 35, 4), ...legR(25, 35, 4), ...arms(15, 8, 75), lift: 0.06 }],
            [0.5, { ...legR(90, 95, 4), ...legL(0, 5, 4), ...armL(60, 8, 70), ...armR(-30, 8, 80), lift: 0.03 }],
            [0.75, { ...legL(25, 35, 4), ...legR(25, 35, 4), ...arms(15, 8, 75), lift: 0.06 }],
        ],
    },

    // ---------- 哑铃 / 壶铃 ----------
    db_shoulder_press: {
        anchor: 'feet',
        props: ['dumbbells'],
        keys: [
            [0, { ...STANDING, armL: limb(10, 80, 1), foreL: [0.1, 1, 0.15], armR: limb(10, 80, -1), foreR: [-0.1, 1, 0.15] }],
            [0.5, { ...STANDING, armL: limb(170, 15, 1), foreL: [0.1, 1, 0.05], armR: limb(170, 15, -1), foreR: [-0.1, 1, 0.05] }],
        ],
    },

    db_curl: {
        anchor: 'feet',
        props: ['dumbbells'],
        keys: [
            [0, { ...STANDING, ...arms(0, 8, 10) }],
            [0.5, { ...STANDING, ...arms(5, 8, 140) }],
        ],
    },

    db_lateral_raise: {
        anchor: 'feet',
        props: ['dumbbells'],
        keys: [
            [0, { ...STANDING, ...arms(0, 10, 10) }],
            [0.5, { ...STANDING, ...arms(0, 85, 10) }],
        ],
    },

    db_rdl: {
        anchor: 'feet',
        props: ['dumbbells'],
        keys: [
            [0, { legsWorld: true, thighL: [0.02, -1, 0.05], shinL: [0, -1, -0.04], thighR: [-0.02, -1, 0.05], shinR: [0, -1, -0.04], ...straightArmsDown(0.12, 0.05) }],
            [0.5, { pitch: 60, head: up(-20), legsWorld: true, thighL: [0.02, -0.95, 0.3], shinL: [0, -1, -0.08], thighR: [-0.02, -0.95, 0.3], shinR: [0, -1, -0.08], ...straightArmsDown(0.1, 0) }],
        ],
    },

    kb_swing: {
        anchor: 'feet',
        props: ['kettlebell'],
        keys: [
            [0, { legsWorld: true, thighL: [0.04, -1, 0], shinL: [0, -1, 0], thighR: [-0.04, -1, 0], shinR: [0, -1, 0], armsWorld: true, armL: v(0.06, 0, 1), foreL: v(0.06, 0, 1), armR: v(-0.06, 0, 1), foreR: v(-0.06, 0, 1) }],
            [0.5, { pitch: 55, head: up(-25), legsWorld: true, thighL: [0.06, -0.9, 0.42], shinL: [0, -1, -0.1], thighR: [-0.06, -0.9, 0.42], shinR: [0, -1, -0.1], armsWorld: true, armL: v(0.06, -0.6, -0.8), foreL: v(0.06, -0.6, -0.8), armR: v(-0.06, -0.6, -0.8), foreR: v(-0.06, -0.6, -0.8) }],
        ],
    },

    // ---------- 拉伸 ----------
    child_pose: {
        cycleSec: 5,
        keys: [
            [0, { pitch: 110, legsWorld: true, thighL: [0.12, -0.55, 0.83], shinL: [0.02, 0.05, -1], thighR: [-0.12, -0.55, 0.83], shinR: [-0.02, 0.05, -1], toeL: [0, -0.25, -1], toeR: [0, -0.25, -1], armsWorld: true, armL: v(0.15, -0.25, 1), foreL: v(0.15, -0.2, 1), armR: v(-0.15, -0.25, 1), foreR: v(-0.15, -0.2, 1) }],
            [0.5, { pitch: 106, legsWorld: true, thighL: [0.12, -0.55, 0.83], shinL: [0.02, 0.05, -1], thighR: [-0.12, -0.55, 0.83], shinR: [-0.02, 0.05, -1], toeL: [0, -0.25, -1], toeR: [0, -0.25, -1], armsWorld: true, armL: v(0.15, -0.3, 1), foreL: v(0.15, -0.2, 1), armR: v(-0.15, -0.3, 1), foreR: v(-0.15, -0.2, 1) }],
        ],
    },
};
