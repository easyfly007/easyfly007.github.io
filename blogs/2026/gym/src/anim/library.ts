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

// 双手叉腰
const HANDS_ON_HIPS: Pose = {
    armL: limb(-15, 40, 1),
    foreL: [-0.55, -0.55, 0.6],
    armR: limb(-15, 40, -1),
    foreR: [0.55, -0.55, 0.6],
};

// 双手抱头（手肘向外）
const HANDS_BEHIND_HEAD: Pose = {
    armL: limb(150, 70, 1),
    foreL: [-0.75, 0.45, -0.45],
    armR: limb(150, 70, -1),
    foreR: [0.75, 0.45, -0.45],
};

// 哑铃/杠铃在肩上（推举起始位）
const RACK_ARMS: Pose = {
    armL: limb(10, 80, 1),
    foreL: [0.1, 1, 0.15],
    armR: limb(10, 80, -1),
    foreR: [-0.1, 1, 0.15],
};

const OVERHEAD_ARMS: Pose = {
    armL: limb(170, 15, 1),
    foreL: [0.1, 1, 0.05],
    armR: limb(170, 15, -1),
    foreR: [-0.1, 1, 0.05],
};

// 四点跪姿：手在肩下、膝在髋下，小腿和脚背贴地
const QUADRUPED: Pose = {
    pitch: 76,
    legsWorld: true,
    thighL: [0.04, -1, 0.02],
    shinL: [0, 0.03, -1],
    thighR: [-0.04, -1, 0.02],
    shinR: [0, 0.03, -1],
    toeL: [0, -0.2, -1],
    toeR: [0, -0.2, -1],
    armsWorld: true,
    armL: [0.05, -1, 0],
    foreL: [0.05, -1, 0],
    armR: [-0.05, -1, 0],
    foreR: [-0.05, -1, 0],
};

// 高位平板（直臂撑地）
const HIGH_PLANK: Pose = { pitch: 64, ...straightArmsDown(0.16, 0.03) };

// 俯卧，双臂前伸
const PRONE: Pose = { pitch: 90, ...arms(180, 12) };

// 仰卧伸直
const SUPINE: Pose = { pitch: -90, ...arms(0, 12) };

// 跑步：一条腿前摆、一条腿后蹬，手臂反向摆
function runPose(front: number, kick: number, armSwing: number, lift: number, left: boolean): Pose {
    const a = left ? { ...legL(front, front + 15, 4), ...legR(-25, kick, 4) } : { ...legR(front, front + 15, 4), ...legL(-25, kick, 4) };
    const b = left ? { ...armR(armSwing, 8, 85), ...armL(-armSwing * 0.6, 8, 85) } : { ...armL(armSwing, 8, 85), ...armR(-armSwing * 0.6, 8, 85) };
    return { ...a, ...b, lift };
}

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

    // ======================= 热身 / 激活 =======================
    arm_circles: {
        cycleSec: 1.6,
        linear: true,
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, armL: limb(12, 88, 1), foreL: limb(12, 88, 1), armR: limb(12, 88, -1), foreR: limb(12, 88, -1) }],
            [0.25, { ...STANDING, armL: limb(0, 100, 1), foreL: limb(0, 100, 1), armR: limb(0, 100, -1), foreR: limb(0, 100, -1) }],
            [0.5, { ...STANDING, armL: limb(-12, 88, 1), foreL: limb(-12, 88, 1), armR: limb(-12, 88, -1), foreR: limb(-12, 88, -1) }],
            [0.75, { ...STANDING, armL: limb(0, 76, 1), foreL: limb(0, 76, 1), armR: limb(0, 76, -1), foreR: limb(0, 76, -1) }],
        ],
    },

    hip_circles: {
        cycleSec: 3,
        linear: true,
        anchor: 'feet',
        keys: [
            [0, { ...legs(0, 8, 8), ...HANDS_ON_HIPS, pitch: 8, spine: up(-8) }],
            [0.25, { ...legs(0, 8, 8), ...HANDS_ON_HIPS, roll: 8, spine: up(0, -8) }],
            [0.5, { ...legs(0, 8, 8), ...HANDS_ON_HIPS, pitch: -8, spine: up(8) }],
            [0.75, { ...legs(0, 8, 8), ...HANDS_ON_HIPS, roll: -8, spine: up(0, 8) }],
        ],
    },

    torso_twist: {
        cycleSec: 2,
        anchor: 'feet',
        keys: [
            [0, { ...legs(0, 5, 8), twist: 50, ...arms(25, 35, 50) }],
            [0.5, { ...legs(0, 5, 8), twist: -50, ...arms(25, 35, 50) }],
        ],
    },

    leg_swings: {
        anchor: 'footR',
        keys: [
            [0, { ...legL(-30, 10, 3), ...legR(0, 0, 3), ...arms(0, 45) }],
            [0.5, { ...legL(65, 5, 3), ...legR(0, 0, 3), ...arms(0, 45) }],
        ],
    },

    march_in_place: {
        cycleSec: 1.2,
        keys: [
            [0, { ...legL(70, 80, 4), ...legR(0, 0, 4), ...armR(45, 8, 80), ...armL(-25, 8, 80) }],
            [0.25, { ...legs(5, 8, 4), ...arms(10, 8, 80) }],
            [0.5, { ...legR(70, 80, 4), ...legL(0, 0, 4), ...armL(45, 8, 80), ...armR(-25, 8, 80) }],
            [0.75, { ...legs(5, 8, 4), ...arms(10, 8, 80) }],
        ],
    },

    inchworm: {
        keys: [
            [0, STANDING],
            [0.2, { pitch: 115, head: up(-20), legsWorld: true, thighL: [0.03, -1, 0.05], shinL: [0, -1, -0.02], thighR: [-0.03, -1, 0.05], shinR: [0, -1, -0.02], ...straightArmsDown(0.12, 0.15) }],
            [0.45, { ...HIGH_PLANK }],
            [0.6, { ...HIGH_PLANK }],
            [0.8, { pitch: 115, head: up(-20), legsWorld: true, thighL: [0.03, -1, 0.05], shinL: [0, -1, -0.02], thighR: [-0.03, -1, 0.05], shinR: [0, -1, -0.02], ...straightArmsDown(0.12, 0.15) }],
        ],
    },

    cat_cow: {
        keys: [
            [0, { ...QUADRUPED, spine: up(-18), head: up(45) }],
            [0.5, { ...QUADRUPED, pitch: 80, spine: up(15), head: up(-35) }],
        ],
    },

    worlds_greatest_stretch: {
        anchor: 'footL',
        keys: [
            [0, { pitch: 55, legsWorld: true, thighL: [0.08, -0.15, 1], shinL: [0.02, -1, -0.12], thighR: [-0.06, -0.55, -0.85], shinR: [-0.06, -0.55, -0.85], toeR: [0, -0.5, 0.85], armsWorld: true, armL: [0.05, -1, 0], foreL: [0.05, -1, 0], armR: [-0.05, -1, 0.05], foreR: [-0.05, -1, 0.05] }],
            [0.5, { pitch: 55, twist: 70, head: up(-10, 30), legsWorld: true, thighL: [0.08, -0.15, 1], shinL: [0.02, -1, -0.12], thighR: [-0.06, -0.55, -0.85], shinR: [-0.06, -0.55, -0.85], toeR: [0, -0.5, 0.85], armsWorld: true, armL: [0.25, 1, -0.1], foreL: [0.25, 1, -0.1], armR: [-0.05, -1, 0.05], foreR: [-0.05, -1, 0.05] }],
        ],
    },

    bw_good_morning: {
        anchor: 'feet',
        keys: [
            [0, { legsWorld: true, thighL: [0.04, -1, 0.05], shinL: [0, -1, -0.03], thighR: [-0.04, -1, 0.05], shinR: [0, -1, -0.03], ...HANDS_BEHIND_HEAD }],
            [0.5, { pitch: 60, legsWorld: true, thighL: [0.04, -0.95, 0.3], shinL: [0, -1, -0.08], thighR: [-0.04, -0.95, 0.3], shinR: [0, -1, -0.08], ...HANDS_BEHIND_HEAD }],
        ],
    },

    jump_rope_easy: {
        cycleSec: 0.7,
        anchor: 'feet',
        props: ['rope'],
        keys: [
            [0, { ...legs(0, 18, 3), ...arms(10, 25, 55) }],
            [0.5, { ...legs(0, 4, 3), ...arms(10, 25, 55), lift: 0.05, toeL: [0, -0.4, 0.9], toeR: [0, -0.4, 0.9] }],
        ],
    },

    light_jog: {
        cycleSec: 0.75,
        keys: [
            [0, runPose(40, 55, 40, 0.02, true)],
            [0.25, { ...legs(12, 25, 4), ...arms(10, 8, 85), lift: 0.04 }],
            [0.5, runPose(40, 55, 40, 0.02, false)],
            [0.75, { ...legs(12, 25, 4), ...arms(10, 8, 85), lift: 0.04 }],
        ],
    },

    band_pull_apart: {
        anchor: 'feet',
        props: ['bandHands'],
        keys: [
            [0, { ...STANDING, ...arms(90, 12) }],
            [0.5, { ...STANDING, armL: limb(5, 88, 1), foreL: limb(5, 88, 1), armR: limb(5, 88, -1), foreR: limb(5, 88, -1) }],
        ],
    },

    kb_halo: {
        anchor: 'feet',
        props: ['kettlebellGoblet'],
        linear: true,
        keys: [
            [0, { ...STANDING, armL: limb(70, 20, 1), foreL: [-0.45, 0.8, 0.35], armR: limb(70, 20, -1), foreR: [0.45, 0.8, 0.35] }],
            [0.25, { ...STANDING, armL: limb(150, 45, 1), foreL: [-0.85, 0.35, -0.3], armR: limb(95, 10, -1), foreR: [-0.1, 0.9, -0.3] }],
            [0.5, { ...STANDING, armL: limb(160, 35, 1), foreL: [-0.6, 0.2, -0.75], armR: limb(160, 35, -1), foreR: [0.6, 0.2, -0.75] }],
            [0.75, { ...STANDING, armL: limb(95, 10, 1), foreL: [0.1, 0.9, -0.3], armR: limb(150, 45, -1), foreR: [0.85, 0.35, -0.3] }],
        ],
    },

    // ======================= 主训练：下肢 =======================
    step_up: {
        anchor: 'footL',
        plant: { part: 'footL', y: 0.45 },
        scene: [{ kind: 'chair', at: 'footL', offset: [0, 0, 0.05] }],
        keys: [
            [0, { ...legL(88, 93, 5), ...legR(0, 0, 5), spine: up(15) }],
            [0.5, { ...legL(0, 0, 5), ...legR(25, 50, 5), spine: up(0) }],
        ],
    },

    bulgarian_split_squat: {
        anchor: 'footL',
        scene: [{ kind: 'chair', at: 'footR', offset: [0, 0, -0.08] }],
        keys: [
            [0, { ...legL(15, 10, 5), ...legR(-25, 65, 5), toeR: [0, -0.6, -0.8] }],
            [0.5, { ...legL(80, 85, 5), ...legR(-37, 96, 5), toeR: [0, -0.6, -0.8], spine: up(8) }],
        ],
    },

    squat_jump: {
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, armsWorld: true, ...arms(0, 10) }],
            [0.3, { ...legs(85, 110, 12), spine: up(35), armsWorld: true, ...arms(-30, 10) }],
            [0.55, { ...legs(0, 0, 6), armsWorld: true, ...arms(160, 15), lift: 0.3, toeL: [0, -0.7, 0.7], toeR: [0, -0.7, 0.7] }],
            [0.75, { ...legs(40, 60, 8), spine: up(15), armsWorld: true, ...arms(30, 10) }],
        ],
    },

    db_lunge: {
        anchor: 'footR',
        props: ['dumbbells'],
        keys: [
            [0, { ...STANDING, ...arms(0, 10) }],
            [0.5, { ...legL(80, 85, 4), ...legR(-15, 85, 4), spine: up(4), ...arms(0, 10) }],
        ],
    },

    db_thruster: {
        anchor: 'feet',
        props: ['dumbbells'],
        keys: [
            [0, { ...STANDING, ...RACK_ARMS }],
            [0.35, { ...squatDown(25), ...RACK_ARMS }],
            [0.7, { ...STANDING, ...OVERHEAD_ARMS }],
        ],
    },

    band_squat: squatDef(squatDown(30), RACK_ARMS, { props: ['bandFeet'] }),

    band_lateral_walk: {
        anchor: 'footR',
        props: ['bandKnees'],
        keys: [
            [0, { ...legs(35, 45, 6), spine: up(20), ...HANDS_ON_HIPS }],
            [0.5, { ...legL(35, 45, 20), ...legR(35, 45, 6), spine: up(20), ...HANDS_ON_HIPS }],
        ],
    },

    bb_deadlift: {
        anchor: 'feet',
        props: ['barbell'],
        keys: [
            [0, { legsWorld: true, thighL: [0.04, -1, 0.04], shinL: [0, -1, -0.03], thighR: [-0.04, -1, 0.04], shinR: [0, -1, -0.03], ...straightArmsDown(0.2, 0.05) }],
            [0.5, { pitch: 50, head: up(-25), legsWorld: true, thighL: [0.06, -0.72, 0.7], shinL: [0, -0.96, -0.28], thighR: [-0.06, -0.72, 0.7], shinR: [0, -0.96, -0.28], ...straightArmsDown(0.2, 0.02) }],
        ],
    },

    // ======================= 主训练：上肢 =======================
    incline_pushup: {
        anchor: 'feet',
        scene: [{ kind: 'chair', at: 'hands' }],
        keys: [
            [0, { pitch: 47, toeL: [0, -0.6, 0.8], toeR: [0, -0.6, 0.8], ...straightArmsDown(0.18, 0.05) }],
            [0.5, { pitch: 58, toeL: [0, -0.6, 0.8], toeR: [0, -0.6, 0.8], armsWorld: true, armL: v(0.55, -0.3, -0.78), foreL: v(0.05, -1, 0.1), armR: v(-0.55, -0.3, -0.78), foreR: v(-0.05, -1, 0.1) }],
        ],
    },

    pike_pushup: {
        anchor: 'feet',
        keys: [
            [0, { pitch: 135, head: up(-10), legsWorld: true, thighL: [0.04, -0.75, -0.66], shinL: [0.04, -0.75, -0.66], thighR: [-0.04, -0.75, -0.66], shinR: [-0.04, -0.75, -0.66], ...straightArmsDown(0.18, 0.1) }],
            [0.5, { pitch: 150, head: up(-10), legsWorld: true, thighL: [0.04, -0.75, -0.66], shinL: [0.04, -0.75, -0.66], thighR: [-0.04, -0.75, -0.66], shinR: [-0.04, -0.75, -0.66], armsWorld: true, armL: v(0.6, -0.2, -0.75), foreL: v(0.05, -1, 0.15), armR: v(-0.6, -0.2, -0.75), foreR: v(-0.05, -1, 0.15) }],
        ],
    },

    chair_dip: {
        anchor: 'hands',
        scene: [{ kind: 'chair', at: 'hands', offset: [0, 0, -0.1] }],
        keys: [
            [0, { ...legs(77, 58, 6), armsWorld: true, armL: v(0.1, -1, -0.28), foreL: v(0.1, -1, -0.28), armR: v(-0.1, -1, -0.28), foreR: v(-0.1, -1, -0.28) }],
            [0.5, { ...legs(109, 99, 6), armsWorld: true, armL: v(0.1, -0.3, -1), foreL: v(0.05, -1, 0.15), armR: v(-0.1, -0.3, -1), foreR: v(-0.05, -1, 0.15) }],
        ],
    },

    pullup: {
        hang: true,
        scene: [{ kind: 'bar', at: 'hands' }],
        keys: [
            [0, { ...legs(10, 25), armsWorld: true, armL: v(0.3, 1, 0), foreL: v(0.15, 1, 0), armR: v(-0.3, 1, 0), foreR: v(-0.15, 1, 0) }],
            [0.5, { ...legs(15, 35), head: up(-15), armsWorld: true, armL: v(0.85, -0.45, 0.1), foreL: v(-0.05, 1, 0), armR: v(-0.85, -0.45, 0.1), foreR: v(0.05, 1, 0) }],
        ],
    },

    hanging_knee_raise: {
        hang: true,
        scene: [{ kind: 'bar', at: 'hands' }],
        keys: [
            [0, { ...legs(5, 10), armsWorld: true, armL: v(0.3, 1, 0), foreL: v(0.15, 1, 0), armR: v(-0.3, 1, 0), foreR: v(-0.15, 1, 0) }],
            [0.5, { ...legs(110, 110), spine: up(-10), armsWorld: true, armL: v(0.3, 1, 0), foreL: v(0.15, 1, 0), armR: v(-0.3, 1, 0), foreR: v(-0.15, 1, 0) }],
        ],
    },

    db_row: {
        anchor: 'feet',
        props: ['dumbbellL'],
        keys: [
            [0, { pitch: 50, head: up(-20), legsWorld: true, thighL: [0.06, -0.95, -0.25], shinL: [0.04, -1, -0.1], thighR: [-0.08, -0.8, 0.6], shinR: [-0.02, -1, -0.12], armsWorld: true, armL: v(0.1, -1, 0.05), foreL: v(0.1, -1, 0.05), armR: v(-0.15, -0.75, 0.6), foreR: v(-0.05, -1, 0.1) }],
            [0.5, { pitch: 50, head: up(-20), legsWorld: true, thighL: [0.06, -0.95, -0.25], shinL: [0.04, -1, -0.1], thighR: [-0.08, -0.8, 0.6], shinR: [-0.02, -1, -0.12], armsWorld: true, armL: v(0.12, -0.15, -1), foreL: v(0.05, -1, 0.05), armR: v(-0.15, -0.75, 0.6), foreR: v(-0.05, -1, 0.1) }],
        ],
    },

    db_floor_press: {
        anchor: 'feet',
        props: ['dumbbells'],
        keys: [
            [0, { ...SUPINE_KNEES_BENT, head: up(10), armsWorld: true, armL: v(0.85, -0.05, -0.5), foreL: v(0, 1, 0), armR: v(-0.85, -0.05, -0.5), foreR: v(0, 1, 0) }],
            [0.5, { ...SUPINE_KNEES_BENT, head: up(10), armsWorld: true, armL: v(0.12, 1, 0.05), foreL: v(0.08, 1, 0.05), armR: v(-0.12, 1, 0.05), foreR: v(-0.08, 1, 0.05) }],
        ],
    },

    bb_bent_row: {
        anchor: 'feet',
        props: ['barbell'],
        keys: [
            [0, { pitch: 50, head: up(-20), legsWorld: true, thighL: [0.05, -0.92, 0.38], shinL: [0, -1, -0.1], thighR: [-0.05, -0.92, 0.38], shinR: [0, -1, -0.1], ...straightArmsDown(0.22, 0.05) }],
            [0.5, { pitch: 50, head: up(-20), legsWorld: true, thighL: [0.05, -0.92, 0.38], shinL: [0, -1, -0.1], thighR: [-0.05, -0.92, 0.38], shinR: [0, -1, -0.1], armsWorld: true, armL: v(0.35, -0.2, -1), foreL: v(0.12, -1, 0.05), armR: v(-0.35, -0.2, -1), foreR: v(-0.12, -1, 0.05) }],
        ],
    },

    bb_overhead_press: {
        anchor: 'feet',
        props: ['barbell'],
        keys: [
            [0, { ...STANDING, armL: limb(25, 55, 1), foreL: [0.05, 1, 0.25], armR: limb(25, 55, -1), foreR: [-0.05, 1, 0.25] }],
            [0.5, { ...STANDING, ...OVERHEAD_ARMS }],
        ],
    },

    band_row: {
        anchor: 'feet',
        props: ['bandFeet'],
        keys: [
            [0, { pitch: 40, head: up(-15), legsWorld: true, thighL: [0.05, -0.85, 0.5], shinL: [0, -1, -0.15], thighR: [-0.05, -0.85, 0.5], shinR: [0, -1, -0.15], armsWorld: true, armL: v(0.12, -1, 0.15), foreL: v(0.12, -1, 0.15), armR: v(-0.12, -1, 0.15), foreR: v(-0.12, -1, 0.15) }],
            [0.5, { pitch: 40, head: up(-15), legsWorld: true, thighL: [0.05, -0.85, 0.5], shinL: [0, -1, -0.15], thighR: [-0.05, -0.85, 0.5], shinR: [0, -1, -0.15], armsWorld: true, armL: v(0.2, -0.2, -1), foreL: v(0.05, -1, 0.1), armR: v(-0.2, -0.2, -1), foreR: v(-0.05, -1, 0.1) }],
        ],
    },

    band_chest_press: {
        anchor: 'feet',
        props: ['bandBack'],
        keys: [
            [0, { ...legL(10, 10, 5), ...legR(-10, 10, 5), armL: limb(-20, 55, 1), foreL: [0.05, 0, 1], armR: limb(-20, 55, -1), foreR: [-0.05, 0, 1] }],
            [0.5, { ...legL(10, 10, 5), ...legR(-10, 10, 5), ...arms(90, 6) }],
        ],
    },

    band_face_pull: {
        anchor: 'feet',
        props: ['bandAnchor'],
        keys: [
            [0, { ...legL(10, 10, 5), ...legR(-10, 10, 5), ...arms(95, 10) }],
            [0.5, { ...legL(10, 10, 5), ...legR(-10, 10, 5), armL: limb(95, 75, 1), foreL: [0.1, 0.75, 0.65], armR: limb(95, 75, -1), foreR: [-0.1, 0.75, 0.65] }],
        ],
    },

    // ======================= 主训练：核心 =======================
    superman: {
        keys: [
            [0, { ...PRONE, toeL: [0, -0.3, -1], toeR: [0, -0.3, -1] }],
            [0.5, { ...PRONE, spine: up(-12), head: up(-15), ...arms(195, 12), ...legs(-15, 0), toeL: [0, -0.3, -1], toeR: [0, -0.3, -1] }],
            [0.75, { ...PRONE, spine: up(-12), head: up(-15), ...arms(195, 12), ...legs(-15, 0), toeL: [0, -0.3, -1], toeR: [0, -0.3, -1] }],
        ],
    },

    side_plank: {
        cycleSec: 4,
        keys: [
            [0, { roll: 72, armsWorld: true, armL: v(0, -1, 0), foreL: v(0, -0.05, 1), armR: v(0, 1, 0), foreR: v(0, 1, 0) }],
            [0.5, { roll: 74, armsWorld: true, armL: v(0, -1, 0), foreL: v(0, -0.05, 1), armR: v(0, 1, 0), foreR: v(0, 1, 0) }],
        ],
    },

    dead_bug: {
        anchor: 'none',
        keys: [
            [0, { ...SUPINE, ...arms(90, 8), ...legs(90, 90) }],
            [0.5, { ...SUPINE, ...armL(170, 8), ...armR(90, 8), ...legL(90, 90), ...legR(12, 0) }],
        ],
    },

    bird_dog: {
        keys: [
            [0, QUADRUPED],
            [0.5, { ...QUADRUPED, armL: [0.05, 0.1, 1], foreL: [0.05, 0.1, 1], thighR: [-0.04, 0.12, -1], shinR: [-0.04, 0.12, -1], toeR: [0, -1, -0.2] }],
        ],
    },

    bicycle_crunch: {
        repsPerCycle: 2,
        keys: [
            [0, { ...SUPINE, spine: up(30), twist: 35, ...HANDS_BEHIND_HEAD, ...legL(105, 95), ...legR(35, 0) }],
            [0.5, { ...SUPINE, spine: up(30), twist: -35, ...HANDS_BEHIND_HEAD, ...legR(105, 95), ...legL(35, 0) }],
        ],
    },

    standing_crunch: {
        repsPerCycle: 2,
        keys: [
            [0, { ...STANDING, ...HANDS_BEHIND_HEAD }],
            [0.25, { ...legL(95, 90, 4), ...legR(0, 0, 4), spine: up(20, 10), twist: 30, ...HANDS_BEHIND_HEAD }],
            [0.5, { ...STANDING, ...HANDS_BEHIND_HEAD }],
            [0.75, { ...legR(95, 90, 4), ...legL(0, 0, 4), spine: up(20, -10), twist: -30, ...HANDS_BEHIND_HEAD }],
        ],
    },

    // ======================= 主训练：有氧 / 爆发 =======================
    burpee: {
        cycleSec: 3,
        keys: [
            [0, STANDING],
            [0.18, { ...legs(100, 135, 12), spine: up(55), ...straightArmsDown(0.18, 0.2) }],
            [0.38, { ...HIGH_PLANK }],
            [0.52, { ...HIGH_PLANK }],
            [0.68, { ...legs(100, 135, 12), spine: up(55), ...straightArmsDown(0.18, 0.2) }],
            [0.84, { ...legs(0, 0, 6), ...arms(170, 15), lift: 0.25, toeL: [0, -0.7, 0.7], toeR: [0, -0.7, 0.7] }],
        ],
    },

    skater: {
        cycleSec: 1.6,
        keys: [
            [0, { dx: -0.45, pitch: 20, ...legR(45, 55, 4), thighL: limb(-15, -18, 1), shinL: limb(-60, -18, 1), ...armL(60, 10, 30), ...armR(-40, 10, 30) }],
            [0.25, { lift: 0.18, pitch: 15, ...legs(20, 30, 6), ...arms(10, 10, 30) }],
            [0.5, { dx: 0.45, pitch: 20, ...legL(45, 55, 4), thighR: limb(-15, -18, -1), shinR: limb(-60, -18, -1), ...armR(60, 10, 30), ...armL(-40, 10, 30) }],
            [0.75, { lift: 0.18, pitch: 15, ...legs(20, 30, 6), ...arms(10, 10, 30) }],
        ],
    },

    jump_rope: {
        cycleSec: 0.5,
        anchor: 'feet',
        props: ['rope'],
        keys: [
            [0, { ...legs(0, 18, 3), ...arms(10, 25, 55) }],
            [0.5, { ...legs(0, 4, 3), ...arms(10, 25, 55), lift: 0.06, toeL: [0, -0.4, 0.9], toeR: [0, -0.4, 0.9] }],
        ],
    },

    sprint: {
        cycleSec: 0.5,
        keys: [
            [0, { ...runPose(75, 100, 70, 0.04, true), pitch: 10 }],
            [0.25, { ...legs(20, 40, 4), ...arms(10, 8, 85), lift: 0.08, pitch: 10 }],
            [0.5, { ...runPose(75, 100, 70, 0.04, false), pitch: 10 }],
            [0.75, { ...legs(20, 40, 4), ...arms(10, 8, 85), lift: 0.08, pitch: 10 }],
        ],
    },

    shadow_boxing: {
        cycleSec: 1.4,
        anchor: 'feet',
        keys: [
            [0, { ...legL(15, 20, 6), ...legR(-15, 20, 6), twist: 10, ...arms(40, 20, 120) }],
            [0.2, { ...legL(15, 20, 6), ...legR(-15, 20, 6), twist: 35, ...armL(40, 20, 120), armR: limb(90, -5, -1), foreR: limb(90, -5, -1) }],
            [0.45, { ...legL(15, 20, 6), ...legR(-15, 20, 6), twist: 10, ...arms(40, 20, 120) }],
            [0.65, { ...legL(15, 20, 6), ...legR(-15, 20, 6), twist: -15, armL: limb(90, -5, 1), foreL: limb(90, -5, 1), ...armR(40, 20, 120) }],
        ],
    },

    // ======================= 放松 / 拉伸 =======================
    standing_quad_stretch: {
        cycleSec: 5,
        anchor: 'footR',
        keys: [
            [0, { ...legR(0, 0, 3), ...legL(-8, 155, 3), toeL: [0, 0.6, -0.8], armL: limb(-35, 15, 1), foreL: limb(-80, 10, 1), ...armR(10, 60) }],
            [0.5, { ...legR(0, 0, 3), ...legL(-12, 158, 3), toeL: [0, 0.6, -0.8], armL: limb(-40, 15, 1), foreL: limb(-85, 10, 1), ...armR(10, 62) }],
        ],
    },

    standing_hamstring_stretch: {
        cycleSec: 5,
        anchor: 'footR',
        keys: [
            [0, { pitch: 30, legsWorld: true, thighL: [0.05, -0.75, 0.66], shinL: [0.05, -0.75, 0.66], thighR: [-0.05, -0.97, 0.25], shinR: [0, -1, -0.1], toeL: [0, 0.6, 0.8], armsWorld: true, armL: v(0.1, -0.8, 0.6), foreL: v(0.1, -0.9, 0.4), armR: v(-0.1, -0.8, 0.6), foreR: v(-0.1, -0.9, 0.4) }],
            [0.5, { pitch: 36, legsWorld: true, thighL: [0.05, -0.75, 0.66], shinL: [0.05, -0.75, 0.66], thighR: [-0.05, -0.97, 0.25], shinR: [0, -1, -0.1], toeL: [0, 0.6, 0.8], armsWorld: true, armL: v(0.1, -0.8, 0.6), foreL: v(0.1, -0.9, 0.45), armR: v(-0.1, -0.8, 0.6), foreR: v(-0.1, -0.9, 0.45) }],
        ],
    },

    calf_stretch: {
        cycleSec: 5,
        anchor: 'footR',
        scene: [{ kind: 'wall', at: 'hands', offset: [0, 0, 0.05] }],
        keys: [
            [0, { pitch: 18, legsWorld: true, thighL: [0.05, -0.85, -0.5], shinL: [0.05, -0.85, -0.5], thighR: [-0.05, -0.85, 0.5], shinR: [0, -1, 0.05], toeL: [0, 0, 1], armsWorld: true, armL: v(0.15, 0.25, 1), foreL: v(0.1, 0.3, 1), armR: v(-0.15, 0.25, 1), foreR: v(-0.1, 0.3, 1) }],
            [0.5, { pitch: 22, legsWorld: true, thighL: [0.05, -0.85, -0.5], shinL: [0.05, -0.85, -0.5], thighR: [-0.05, -0.85, 0.5], shinR: [0, -1, 0.05], toeL: [0, 0, 1], armsWorld: true, armL: v(0.15, 0.25, 1), foreL: v(0.1, 0.25, 1), armR: v(-0.15, 0.25, 1), foreR: v(-0.1, 0.25, 1) }],
        ],
    },

    figure_four_stretch: {
        cycleSec: 5,
        anchor: 'footR',
        keys: [
            [0, { ...legR(70, 85, 3), thighL: limb(75, 45, 1), shinL: [-0.95, -0.1, 0.3], toeL: [-0.3, 0, 1], spine: up(30), ...arms(80, 10) }],
            [0.5, { ...legR(75, 92, 3), thighL: limb(78, 45, 1), shinL: [-0.95, -0.1, 0.3], toeL: [-0.3, 0, 1], spine: up(34), ...arms(80, 10) }],
        ],
    },

    chest_wall_stretch: {
        cycleSec: 5,
        anchor: 'feet',
        scene: [{ kind: 'wall', at: 'handL', offset: [0.04, 0, 0], turn: true }],
        keys: [
            [0, { ...STANDING, twist: -15, armL: limb(0, 92, 1), foreL: limb(0, 92, 1), ...armR(0, 8) }],
            [0.5, { ...STANDING, twist: -25, armL: limb(-10, 92, 1), foreL: limb(-10, 92, 1), ...armR(0, 8) }],
        ],
    },

    cross_body_shoulder_stretch: {
        cycleSec: 5,
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, armL: [-0.85, -0.05, 0.5], foreL: [-0.95, -0.05, 0.3], armR: limb(30, 10, -1), foreR: [0.75, 0.35, 0.55] }],
            [0.5, { ...STANDING, armL: [-0.9, -0.05, 0.42], foreL: [-0.97, -0.05, 0.22], armR: limb(32, 10, -1), foreR: [0.75, 0.35, 0.55] }],
        ],
    },

    triceps_stretch: {
        cycleSec: 5,
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, armL: limb(170, 8, 1), foreL: [-0.35, -0.75, -0.55], armR: limb(150, 35, -1), foreR: [0.85, 0.4, -0.2] }],
            [0.5, { ...STANDING, armL: limb(175, 6, 1), foreL: [-0.38, -0.75, -0.55], armR: limb(152, 35, -1), foreR: [0.85, 0.4, -0.2] }],
        ],
    },

    standing_side_bend: {
        cycleSec: 5,
        anchor: 'feet',
        keys: [
            [0, { ...legs(0, 0, 6), spine: up(0, -20), armL: limb(170, 30, 1), foreL: limb(170, 40, 1), ...armR(0, 10) }],
            [0.5, { ...legs(0, 0, 6), spine: up(0, -28), armL: limb(172, 34, 1), foreL: limb(172, 46, 1), ...armR(0, 10) }],
        ],
    },

    deep_breathing: {
        cycleSec: 10,
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, ...arms(0, 8) }],
            [0.4, { ...STANDING, spine: up(-3), head: up(-5), ...arms(0, 22) }],
        ],
    },

    downward_dog: {
        cycleSec: 5,
        anchor: 'feet',
        keys: [
            [0, { pitch: 140, head: up(15), legsWorld: true, thighL: [0.05, -0.8, -0.6], shinL: [0.05, -0.8, -0.6], thighR: [-0.05, -0.8, -0.6], shinR: [-0.05, -0.8, -0.6], toeL: [0, -0.3, 0.95], toeR: [0, -0.3, 0.95], armsWorld: true, armL: v(0.12, -0.72, 0.68), foreL: v(0.12, -0.72, 0.68), armR: v(-0.12, -0.72, 0.68), foreR: v(-0.12, -0.72, 0.68) }],
            [0.5, { pitch: 143, head: up(15), legsWorld: true, thighL: [0.05, -0.82, -0.57], shinL: [0.05, -0.82, -0.57], thighR: [-0.05, -0.82, -0.57], shinR: [-0.05, -0.82, -0.57], toeL: [0, -0.2, 0.98], toeR: [0, -0.2, 0.98], armsWorld: true, armL: v(0.12, -0.72, 0.68), foreL: v(0.12, -0.72, 0.68), armR: v(-0.12, -0.72, 0.68), foreR: v(-0.12, -0.72, 0.68) }],
        ],
    },

    cobra: {
        cycleSec: 5,
        keys: [
            [0, { pitch: 90, spine: up(-50), head: up(-20), toeL: [0, -0.3, -1], toeR: [0, -0.3, -1], armsWorld: true, armL: v(0.2, -1, 0.15), foreL: v(0.1, -1, 0.25), armR: v(-0.2, -1, 0.15), foreR: v(-0.1, -1, 0.25) }],
            [0.5, { pitch: 90, spine: up(-55), head: up(-25), toeL: [0, -0.3, -1], toeR: [0, -0.3, -1], armsWorld: true, armL: v(0.2, -1, 0.1), foreL: v(0.1, -1, 0.2), armR: v(-0.2, -1, 0.1), foreR: v(-0.1, -1, 0.2) }],
        ],
    },

    pigeon_pose: {
        cycleSec: 5,
        keys: [
            [0, { pitch: 35, legsWorld: true, thighL: [0.45, -0.3, 0.85], shinL: [-0.95, -0.05, 0.25], thighR: [-0.05, -0.2, -1], shinR: [-0.05, -0.05, -1], toeL: [-0.3, 0, 1], toeR: [0, -0.3, -1], armsWorld: true, armL: v(0.25, -1, 0.3), foreL: v(0.2, -1, 0.3), armR: v(-0.25, -1, 0.3), foreR: v(-0.2, -1, 0.3) }],
            [0.5, { pitch: 55, legsWorld: true, thighL: [0.45, -0.3, 0.85], shinL: [-0.95, -0.05, 0.25], thighR: [-0.05, -0.2, -1], shinR: [-0.05, -0.05, -1], toeL: [-0.3, 0, 1], toeR: [0, -0.3, -1], armsWorld: true, armL: v(0.25, -0.8, 0.6), foreL: v(0.2, -0.6, 0.8), armR: v(-0.25, -0.8, 0.6), foreR: v(-0.2, -0.6, 0.8) }],
        ],
    },

    seated_forward_fold: {
        cycleSec: 5,
        keys: [
            [0, { pitch: 55, head: up(15), legsWorld: true, thighL: [0.04, -0.05, 1], shinL: [0.04, -0.05, 1], thighR: [-0.04, -0.05, 1], shinR: [-0.04, -0.05, 1], toeL: [0, 1, 0.2], toeR: [0, 1, 0.2], armsWorld: true, armL: v(0.12, -0.45, 1), foreL: v(0.1, -0.35, 1), armR: v(-0.12, -0.45, 1), foreR: v(-0.1, -0.35, 1) }],
            [0.5, { pitch: 65, head: up(15), legsWorld: true, thighL: [0.04, -0.05, 1], shinL: [0.04, -0.05, 1], thighR: [-0.04, -0.05, 1], shinR: [-0.04, -0.05, 1], toeL: [0, 1, 0.2], toeR: [0, 1, 0.2], armsWorld: true, armL: v(0.12, -0.4, 1), foreL: v(0.1, -0.3, 1), armR: v(-0.12, -0.4, 1), foreR: v(-0.1, -0.3, 1) }],
        ],
    },

    supine_twist: {
        cycleSec: 5,
        keys: [
            [0, { ...SUPINE, ...arms(0, 90), legsWorld: true, thighL: [-0.85, -0.05, 0.5], shinL: [-0.05, -0.08, 1], thighR: [0, 0, 1], shinR: [0, 0, 1], toeR: [0, 1, 0.2] }],
            [0.5, { ...SUPINE, ...arms(0, 90), legsWorld: true, thighL: [-0.88, -0.08, 0.45], shinL: [-0.05, -0.08, 1], thighR: [0, 0, 1], shinR: [0, 0, 1], toeR: [0, 1, 0.2] }],
        ],
    },

    kneeling_hip_flexor_stretch: {
        cycleSec: 5,
        keys: [
            [0, { legsWorld: true, thighL: [0.06, -0.1, 1], shinL: [0, -1, 0.05], thighR: [-0.06, -0.95, -0.3], shinR: [0, 0.03, -1], toeR: [0, -0.2, -1], ...HANDS_ON_HIPS }],
            [0.5, { dz: 0.04, legsWorld: true, thighL: [0.06, -0.2, 0.98], shinL: [0, -1, 0.15], thighR: [-0.06, -0.88, -0.45], shinR: [0, 0.03, -1], toeR: [0, -0.2, -1], ...HANDS_ON_HIPS }],
        ],
    },

    warrior_two: {
        cycleSec: 5,
        keys: [
            [0, { legsWorld: true, thighL: [0.75, -0.66, 0], shinL: [0.03, -1, 0], thighR: [-0.5, -0.87, 0], shinR: [-0.5, -0.87, 0], toeL: [1, 0, 0.1], armL: limb(0, 90, 1), foreL: limb(0, 90, 1), armR: limb(0, 90, -1), foreR: limb(0, 90, -1) }],
            [0.5, { legsWorld: true, thighL: [0.78, -0.62, 0], shinL: [0.03, -1, 0], thighR: [-0.5, -0.87, 0], shinR: [-0.5, -0.87, 0], toeL: [1, 0, 0.1], armL: limb(0, 92, 1), foreL: limb(0, 92, 1), armR: limb(0, 92, -1), foreR: limb(0, 92, -1) }],
        ],
    },

    deep_squat_hold: {
        cycleSec: 5,
        anchor: 'feet',
        keys: [
            [0, { ...legs(115, 150, 22), spine: up(25), ...arms(35, 25, 100) }],
            [0.5, { ...legs(117, 152, 22), spine: up(22), ...arms(35, 25, 100) }],
        ],
    },

    // ======================= 扩充：瑜伽垫 / 徒手 =======================
    side_lying_leg_raise: {
        keys: [
            [0, { roll: -90, ...legs(0, 0, 2), armR: limb(180, 10, -1), foreR: [0.5, 0.6, 0], ...armL(35, 0) }],
            [0.5, { roll: -90, ...legL(0, 0, 45), ...legR(0, 0, 2), armR: limb(180, 10, -1), foreR: [0.5, 0.6, 0], ...armL(35, 0) }],
        ],
    },

    clamshell: {
        keys: [
            [0, { roll: -90, pitch: 10, ...legs(50, 90, 2), armR: limb(180, 10, -1), foreR: [0.5, 0.6, 0], ...armL(35, 0) }],
            [0.5, { roll: -90, pitch: 10, thighL: limb(50, 40, 1), shinL: [-0.3, -0.35, -0.85], ...legR(50, 90, 2), armR: limb(180, 10, -1), foreR: [0.5, 0.6, 0], ...armL(35, 0) }],
        ],
    },

    single_leg_glute_bridge: {
        anchor: 'footR',
        keys: [
            [0, { ...SUPINE_KNEES_BENT, thighL: [0.03, 0.73, 0.68], shinL: [0.03, 0.73, 0.68], toeL: [0, 0.6, 0.8], head: up(30), armsWorld: true, armL: v(0.15, -0.05, 1), foreL: v(0.15, -0.05, 1), armR: v(-0.15, -0.05, 1), foreR: v(-0.15, -0.05, 1) }],
            [0.5, {
                ...SUPINE_KNEES_BENT,
                pitch: -122,
                thighL: [0.03, 0.18, 0.98],
                shinL: [0.03, 0.18, 0.98],
                toeL: [0, 0.98, -0.2],
                thighR: [-0.03, 0.18, 0.98],
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

    russian_twist: {
        repsPerCycle: 2,
        keys: [
            [0, { pitch: -40, twist: 45, legsWorld: true, thighL: [0.08, 0.5, 0.86], shinL: [0.04, -0.45, 0.89], thighR: [-0.08, 0.5, 0.86], shinR: [-0.04, -0.45, 0.89], ...arms(70, 5, 40) }],
            [0.5, { pitch: -40, twist: -45, legsWorld: true, thighL: [0.08, 0.5, 0.86], shinL: [0.04, -0.45, 0.89], thighR: [-0.08, 0.5, 0.86], shinR: [-0.04, -0.45, 0.89], ...arms(70, 5, 40) }],
        ],
    },

    v_up: {
        keys: [
            [0, { pitch: -90, legsWorld: true, thighL: [0.04, 0, 1], shinL: [0.04, 0, 1], thighR: [-0.04, 0, 1], shinR: [-0.04, 0, 1], toeL: [0, 1, 0.3], toeR: [0, 1, 0.3], armsWorld: true, armL: v(0.12, 0, -1), foreL: v(0.12, 0, -1), armR: v(-0.12, 0, -1), foreR: v(-0.12, 0, -1) }],
            [0.5, { pitch: -45, legsWorld: true, thighL: [0.04, 0.7, 0.7], shinL: [0.04, 0.7, 0.7], thighR: [-0.04, 0.7, 0.7], shinR: [-0.04, 0.7, 0.7], toeL: [0, 0.7, -0.7], toeR: [0, 0.7, -0.7], armsWorld: true, armL: v(0.08, 0.55, 0.83), foreL: v(0.06, 0.55, 0.83), armR: v(-0.08, 0.55, 0.83), foreR: v(-0.06, 0.55, 0.83) }],
        ],
    },

    reverse_crunch: {
        keys: [
            [0, { ...SUPINE, ...legs(90, 90) }],
            [0.5, { ...SUPINE, pitch: -72, spine: up(-18), ...legs(135, 115) }],
        ],
    },

    leg_raise: {
        keys: [
            [0, { ...SUPINE, ...legs(8, 0) }],
            [0.5, { ...SUPINE, ...legs(88, 0) }],
        ],
    },

    flutter_kicks: {
        cycleSec: 0.7,
        keys: [
            [0, { ...SUPINE, head: up(25), ...legL(22, 0), ...legR(8, 0) }],
            [0.5, { ...SUPINE, head: up(25), ...legL(8, 0), ...legR(22, 0) }],
        ],
    },

    hollow_hold: {
        cycleSec: 4,
        keys: [
            [0, { ...SUPINE, spine: up(22), head: up(10), ...arms(165, 8), ...legs(15, 0) }],
            [0.5, { ...SUPINE, spine: up(20), head: up(10), ...arms(167, 8), ...legs(16, 0) }],
        ],
    },

    bear_plank: {
        cycleSec: 4,
        keys: [
            [0, { ...QUADRUPED, toeL: [0, -0.9, 0.4], toeR: [0, -0.9, 0.4], thighL: [0.04, -1, 0.05], thighR: [-0.04, -1, 0.05] }],
            [0.5, { ...QUADRUPED, pitch: 78, toeL: [0, -0.9, 0.4], toeR: [0, -0.9, 0.4], thighL: [0.04, -1, 0.05], thighR: [-0.04, -1, 0.05] }],
        ],
    },

    plank_jacks: {
        cycleSec: 1,
        anchor: 'hands',
        keys: [
            [0, { ...HIGH_PLANK, ...legs(0, 0, 3) }],
            [0.25, { ...HIGH_PLANK, ...legs(0, 0, 9), lift: 0.04 }],
            [0.5, { ...HIGH_PLANK, ...legs(0, 0, 15) }],
            [0.75, { ...HIGH_PLANK, ...legs(0, 0, 9), lift: 0.04 }],
        ],
    },

    plank_shoulder_tap: {
        cycleSec: 2,
        anchor: 'feet',
        keys: [
            [0, { ...HIGH_PLANK, ...legs(0, 0, 8) }],
            [0.25, { ...HIGH_PLANK, ...legs(0, 0, 8), armL: v(-0.25, -0.35, -0.2), foreL: v(-0.7, 0.6, 0.1) }],
            [0.5, { ...HIGH_PLANK, ...legs(0, 0, 8) }],
            [0.75, { ...HIGH_PLANK, ...legs(0, 0, 8), armR: v(0.25, -0.35, -0.2), foreR: v(0.7, 0.6, 0.1) }],
        ],
    },

    donkey_kick: {
        keys: [
            [0, QUADRUPED],
            [0.5, { ...QUADRUPED, thighL: [0.04, 0.15, -1], shinL: [0.04, 1, 0], toeL: [0, 0, -1] }],
        ],
    },

    fire_hydrant: {
        keys: [
            [0, QUADRUPED],
            [0.5, { ...QUADRUPED, thighL: [1, -0.1, 0], shinL: [0.1, 0, -1], toeL: [0, -0.2, -1] }],
        ],
    },

    reverse_lunge: {
        anchor: 'footL',
        keys: [
            [0, STANDING],
            [0.5, { ...legL(80, 85, 4), ...legR(-15, 85, 4), spine: up(4) }],
        ],
    },

    sumo_squat: squatDef({ ...legs(95, 125, 28), spine: up(15), toeL: [0.4, 0, 0.9], toeR: [-0.4, 0, 0.9] }, { ...legs(0, 0, 22), toeL: [0.4, 0, 0.9], toeR: [-0.4, 0, 0.9], ...arms(30, 10, 90) }),

    wall_pushup: {
        anchor: 'feet',
        scene: [{ kind: 'wall', at: 'hands', offset: [0, 0, 0.05] }],
        keys: [
            [0, { pitch: 20, armsWorld: true, armL: v(0.15, 0.05, 1), foreL: v(0.15, 0.05, 1), armR: v(-0.15, 0.05, 1), foreR: v(-0.15, 0.05, 1) }],
            [0.5, { pitch: 30, armsWorld: true, armL: v(0.6, -0.15, -0.2), foreL: v(0.05, 0.25, 1), armR: v(-0.6, -0.15, -0.2), foreR: v(-0.05, 0.25, 1) }],
        ],
    },

    diamond_pushup: {
        anchor: 'feet',
        keys: [
            [0, { pitch: 64, armsWorld: true, armL: v(-0.12, -1, 0.05), foreL: v(-0.1, -1, 0.05), armR: v(0.12, -1, 0.05), foreR: v(0.1, -1, 0.05) }],
            [0.5, { pitch: 80, armsWorld: true, armL: v(0.2, -0.25, -0.95), foreL: v(-0.15, -1, 0.25), armR: v(-0.2, -0.25, -0.95), foreR: v(0.15, -1, 0.25) }],
        ],
    },

    single_leg_rdl: {
        anchor: 'footL',
        keys: [
            [0, { legsWorld: true, thighL: [0.03, -1, 0.02], shinL: [0, -1, -0.02], thighR: [-0.03, -1, 0.02], shinR: [0, -1, -0.02], ...straightArmsDown(0.12, 0.03) }],
            [0.5, { pitch: 80, head: up(-15), legsWorld: true, thighL: [0.03, -1, 0.1], shinL: [0, -1, -0.05], thighR: [-0.03, 0.17, -0.98], shinR: [-0.03, 0.17, -0.98], toeR: [0, -1, -0.1], ...straightArmsDown(0.1, 0) }],
        ],
    },

    side_lunge: {
        anchor: 'footR',
        keys: [
            [0, { ...legs(0, 0, 6) }],
            [0.5, { pitch: 25, legsWorld: true, thighL: [0.35, -0.55, 0.75], shinL: [0.08, -1, -0.15], thighR: [-0.6, -0.8, 0], shinR: [-0.6, -0.8, 0], toeL: [0.3, 0, 1], toeR: [-0.3, 0, 1], ...arms(70, 10, 60) }],
        ],
    },

    curtsy_lunge: {
        anchor: 'footL',
        keys: [
            [0, STANDING],
            [0.5, { ...legL(75, 80, 3), thighR: limb(-20, -25, -1), shinR: limb(-105, -25, -1), spine: up(5), ...arms(40, 15, 60) }],
        ],
    },

    // ======================= 扩充：壶铃 / 单杠 / 跳绳 / 弹力带 / 哑铃 =======================
    kb_deadlift: {
        anchor: 'feet',
        props: ['kettlebell'],
        keys: [
            [0, { legsWorld: true, thighL: [0.06, -1, 0.04], shinL: [0, -1, -0.03], thighR: [-0.06, -1, 0.04], shinR: [0, -1, -0.03], ...straightArmsDown(0.04, 0.05) }],
            [0.5, { pitch: 55, head: up(-25), legsWorld: true, thighL: [0.08, -0.75, 0.66], shinL: [0, -0.97, -0.24], thighR: [-0.08, -0.75, 0.66], shinR: [0, -0.97, -0.24], ...straightArmsDown(0.04, 0) }],
        ],
    },

    kb_press: {
        anchor: 'feet',
        props: ['kettlebellL'],
        keys: [
            [0, { ...STANDING, armL: limb(20, 30, 1), foreL: [-0.1, 1, 0.1], ...armR(0, 30) }],
            [0.5, { ...STANDING, armL: limb(172, 8, 1), foreL: [0.05, 1, 0], ...armR(0, 30) }],
        ],
    },

    kb_row: {
        anchor: 'feet',
        props: ['kettlebellL'],
        keys: [
            [0, { pitch: 50, head: up(-20), legsWorld: true, thighL: [0.06, -0.95, -0.25], shinL: [0.04, -1, -0.1], thighR: [-0.08, -0.8, 0.6], shinR: [-0.02, -1, -0.12], armsWorld: true, armL: v(0.1, -1, 0.05), foreL: v(0.1, -1, 0.05), armR: v(-0.15, -0.75, 0.6), foreR: v(-0.05, -1, 0.1) }],
            [0.5, { pitch: 50, head: up(-20), legsWorld: true, thighL: [0.06, -0.95, -0.25], shinL: [0.04, -1, -0.1], thighR: [-0.08, -0.8, 0.6], shinR: [-0.02, -1, -0.12], armsWorld: true, armL: v(0.12, -0.15, -1), foreL: v(0.05, -1, 0.05), armR: v(-0.15, -0.75, 0.6), foreR: v(-0.05, -1, 0.1) }],
        ],
    },

    kb_goblet_reverse_lunge: {
        anchor: 'footL',
        props: ['kettlebellGoblet'],
        keys: [
            [0, { ...STANDING, ...GOBLET_ARMS }],
            [0.5, { ...legL(80, 85, 4), ...legR(-15, 85, 4), spine: up(4), ...GOBLET_ARMS }],
        ],
    },

    chin_up: {
        hang: true,
        scene: [{ kind: 'bar', at: 'hands' }],
        keys: [
            [0, { ...legs(10, 25), armsWorld: true, armL: v(0.12, 1, 0), foreL: v(0.05, 1, 0), armR: v(-0.12, 1, 0), foreR: v(-0.05, 1, 0) }],
            [0.5, { ...legs(15, 35), head: up(-15), armsWorld: true, armL: v(0.35, -0.6, 0.7), foreL: v(-0.05, 1, -0.1), armR: v(-0.35, -0.6, 0.7), foreR: v(0.05, 1, -0.1) }],
        ],
    },

    dead_hang: {
        hang: true,
        cycleSec: 4,
        scene: [{ kind: 'bar', at: 'hands' }],
        keys: [
            [0, { ...legs(3, 8), armsWorld: true, armL: v(0.3, 1, 0), foreL: v(0.15, 1, 0), armR: v(-0.3, 1, 0), foreR: v(-0.15, 1, 0) }],
            [0.5, { ...legs(5, 10), armsWorld: true, armL: v(0.3, 1, 0), foreL: v(0.15, 1, 0), armR: v(-0.3, 1, 0), foreR: v(-0.15, 1, 0) }],
        ],
    },

    negative_pullup: {
        hang: true,
        scene: [{ kind: 'bar', at: 'hands' }],
        keys: [
            [0, { ...legs(15, 35), head: up(-15), armsWorld: true, armL: v(0.85, -0.45, 0.1), foreL: v(-0.05, 1, 0), armR: v(-0.85, -0.45, 0.1), foreR: v(0.05, 1, 0) }],
            [0.15, { ...legs(15, 35), head: up(-15), armsWorld: true, armL: v(0.85, -0.45, 0.1), foreL: v(-0.05, 1, 0), armR: v(-0.85, -0.45, 0.1), foreR: v(0.05, 1, 0) }],
            [0.85, { ...legs(10, 25), armsWorld: true, armL: v(0.3, 1, 0), foreL: v(0.15, 1, 0), armR: v(-0.3, 1, 0), foreR: v(-0.15, 1, 0) }],
        ],
    },

    jump_rope_alt: {
        cycleSec: 0.7,
        ropeTurns: 2,
        anchor: 'feet',
        props: ['rope'],
        keys: [
            [0, { ...legL(35, 55, 3), ...legR(0, 8, 3), ...arms(10, 25, 55), lift: 0.03 }],
            [0.5, { ...legR(35, 55, 3), ...legL(0, 8, 3), ...arms(10, 25, 55), lift: 0.03 }],
        ],
    },

    jump_rope_side: {
        cycleSec: 1,
        ropeTurns: 2,
        props: ['rope'],
        keys: [
            [0, { dx: -0.12, ...legs(0, 18, 3), ...arms(10, 25, 55) }],
            [0.25, { ...legs(0, 4, 3), ...arms(10, 25, 55), lift: 0.06 }],
            [0.5, { dx: 0.12, ...legs(0, 18, 3), ...arms(10, 25, 55) }],
            [0.75, { ...legs(0, 4, 3), ...arms(10, 25, 55), lift: 0.06 }],
        ],
    },

    jump_rope_high_knee: {
        cycleSec: 0.8,
        ropeTurns: 2,
        props: ['rope'],
        keys: [
            [0, { ...legL(85, 95, 4), ...legR(0, 5, 4), ...arms(10, 25, 55), lift: 0.04 }],
            [0.5, { ...legR(85, 95, 4), ...legL(0, 5, 4), ...arms(10, 25, 55), lift: 0.04 }],
        ],
    },

    band_curl: {
        anchor: 'feet',
        props: ['bandFeet'],
        keys: [
            [0, { ...STANDING, ...arms(0, 8, 10) }],
            [0.5, { ...STANDING, ...arms(5, 8, 140) }],
        ],
    },

    band_overhead_press: {
        anchor: 'feet',
        props: ['bandFeet'],
        keys: [
            [0, { ...STANDING, ...RACK_ARMS }],
            [0.5, { ...STANDING, ...OVERHEAD_ARMS }],
        ],
    },

    db_reverse_fly: {
        anchor: 'feet',
        props: ['dumbbells'],
        keys: [
            [0, { pitch: 50, head: up(-20), legsWorld: true, thighL: [0.05, -0.92, 0.38], shinL: [0, -1, -0.1], thighR: [-0.05, -0.92, 0.38], shinR: [0, -1, -0.1], armsWorld: true, armL: v(0.12, -1, 0), foreL: v(0.15, -1, 0), armR: v(-0.12, -1, 0), foreR: v(-0.15, -1, 0) }],
            [0.5, { pitch: 50, head: up(-20), legsWorld: true, thighL: [0.05, -0.92, 0.38], shinL: [0, -1, -0.1], thighR: [-0.05, -0.92, 0.38], shinR: [0, -1, -0.1], armsWorld: true, armL: v(1, -0.25, 0), foreL: v(1, -0.4, 0), armR: v(-1, -0.25, 0), foreR: v(-1, -0.4, 0) }],
        ],
    },

    db_triceps_extension: {
        anchor: 'feet',
        props: ['dumbbellGoblet'],
        keys: [
            [0, { ...STANDING, armL: limb(170, 8, 1), foreL: [-0.15, 1, 0], armR: limb(170, 8, -1), foreR: [0.15, 1, 0] }],
            [0.5, { ...STANDING, armL: limb(170, 8, 1), foreL: [-0.2, -0.8, -0.55], armR: limb(170, 8, -1), foreR: [0.2, -0.8, -0.55] }],
        ],
    },

    db_front_raise: {
        anchor: 'feet',
        props: ['dumbbells'],
        keys: [
            [0, { ...STANDING, ...arms(0, 8, 5) }],
            [0.5, { ...STANDING, ...arms(90, 8, 5) }],
        ],
    },

    // ======================= 扩充：拉伸 / 放松 =======================
    neck_stretch: {
        cycleSec: 5,
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, head: up(0, 30), armL: limb(150, 75, 1), foreL: [-0.55, 0.6, 0.2], ...armR(0, 12) }],
            [0.5, { ...STANDING, head: up(0, 36), armL: limb(150, 75, 1), foreL: [-0.6, 0.55, 0.2], ...armR(0, 14) }],
        ],
    },

    shoulder_rolls: {
        cycleSec: 2.5,
        linear: true,
        anchor: 'feet',
        keys: [
            [0, { ...STANDING, ...arms(0, 10) }],
            [0.25, { ...STANDING, ...arms(6, 14), head: up(-4) }],
            [0.5, { ...STANDING, ...arms(-8, 12), spine: up(-3) }],
            [0.75, { ...STANDING, ...arms(-4, 6) }],
        ],
    },

    thread_the_needle: {
        cycleSec: 5,
        keys: [
            [0, { ...QUADRUPED, pitch: 85, twist: -40, armL: [-1, -0.25, 0.1], foreL: [-1, -0.1, 0.1] }],
            [0.5, { ...QUADRUPED, pitch: 87, twist: -45, armL: [-1, -0.25, 0.1], foreL: [-1, -0.1, 0.1] }],
        ],
    },

    butterfly_stretch: {
        cycleSec: 5,
        keys: [
            [0, { pitch: 5, legsWorld: true, thighL: [0.75, -0.25, 0.6], shinL: [-0.95, -0.05, 0.25], thighR: [-0.75, -0.25, 0.6], shinR: [0.95, -0.05, 0.25], toeL: [-0.3, 0, 1], toeR: [0.3, 0, 1], armsWorld: true, armL: v(0.1, -0.6, 0.8), foreL: v(0.05, -0.7, 0.7), armR: v(-0.1, -0.6, 0.8), foreR: v(-0.05, -0.7, 0.7) }],
            [0.5, { pitch: 15, legsWorld: true, thighL: [0.78, -0.3, 0.55], shinL: [-0.95, -0.05, 0.25], thighR: [-0.78, -0.3, 0.55], shinR: [0.95, -0.05, 0.25], toeL: [-0.3, 0, 1], toeR: [0.3, 0, 1], armsWorld: true, armL: v(0.1, -0.6, 0.8), foreL: v(0.05, -0.7, 0.7), armR: v(-0.1, -0.6, 0.8), foreR: v(-0.05, -0.7, 0.7) }],
        ],
    },

    knee_to_chest: {
        cycleSec: 5,
        keys: [
            [0, { ...SUPINE, ...legs(125, 135), ...arms(55, 12, 55) }],
            [0.5, { ...SUPINE, ...legs(132, 138), ...arms(60, 12, 60) }],
        ],
    },
};
