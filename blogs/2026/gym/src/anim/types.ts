// 3D 人偶动作动画的数据结构。
// 坐标约定：x = 人物的左侧，y = 上，z = 人物面朝的前方；长度单位米，角度单位度

export type V3 = [number, number, number];

// 一个时刻的全身姿态。没写的部位用站立默认值
export interface Pose {
    // 整个身体（以骨盆为中心）的朝向：pitch 前倾为正（90 = 俯卧，-90 = 仰卧），roll 向左侧倒为正，yaw 向左转为正
    pitch?: number;
    roll?: number;
    yaw?: number;
    lift?: number;              // 离地高度（跳跃腾空）
    dx?: number;                // 整个人在水平面上平移（世界坐标，米），如滑冰跳左右跳
    dz?: number;
    spine?: V3;                 // 胸腔朝上的方向（骨盆坐标系）
    twist?: number;             // 上身相对骨盆的扭转，向左转为正
    head?: V3;                  // 头顶方向（胸腔坐标系）
    armL?: V3;                  // 上臂方向（胸腔坐标系；armsWorld 时为世界坐标）
    foreL?: V3;                 // 前臂方向（同上）
    armR?: V3;
    foreR?: V3;
    thighL?: V3;                // 大腿方向（骨盆坐标系；legsWorld 时为世界坐标）
    shinL?: V3;
    thighR?: V3;
    shinR?: V3;
    toeL?: V3;                  // 脚尖方向（世界坐标）；不写时自动：站姿脚掌放平，躺/撑时脚尖顺着身体
    toeR?: V3;
    armsWorld?: boolean;
    legsWorld?: boolean;
}

// 跟随身体的器械：每帧按手、脚、肩的位置摆放
export type BodyProp =
    | 'dumbbells'               // 两手各一只
    | 'dumbbellL'               // 只有左手
    | 'dumbbellR'               // 只有右手（左侧动作镜像得到）
    | 'dumbbellGoblet'          // 双手托一只在胸前（竖直）
    | 'kettlebell'              // 双手握一只，沿手臂方向挂在手下（摆荡）
    | 'kettlebellGoblet'        // 双手托一只在胸前，壶身朝下
    | 'kettlebellL'             // 左手单手持壶铃（壶身沿前臂方向）
    | 'kettlebellR'
    | 'barbell'                 // 双手握杠
    | 'barbellBack'             // 杠铃扛在斜方肌上
    | 'bandHands'               // 两手之间拉一根弹力带
    | 'bandKnees'               // 弹力带套在两膝
    | 'bandFeet'                // 两手各拉一根弹力带到同侧脚下
    | 'bandAnchor'              // 两手拉向前方固定点
    | 'bandBack'                // 弹力带从两手绕过后背（推胸）
    | 'rope';                   // 跳绳

// 固定在场景里的物体：位置在第一帧按身体某处算出后就不再移动
export interface SceneProp {
    kind: 'chair' | 'bar' | 'wall';
    at: 'hands' | 'handL' | 'footL' | 'footR' | 'back' | 'front';
    offset?: V3;                // 在算出的位置上再平移（世界坐标）
    turn?: boolean;             // 墙面转 90 度（立在人物侧面）
}

export type Anchor = 'feet' | 'footL' | 'footR' | 'hands' | 'knees' | 'none';

export type ContactPart = 'footL' | 'footR' | 'handL' | 'handR';

// 接触点：这段时间里该部位撑在地上（或凳上）不动，水平位置固定，由膝/肘弯曲来适应。
// from > to 表示跨过循环首尾（如 0.85 → 0.15）；不写 from/to 表示整个循环
export interface Contact {
    part: ContactPart;
    from?: number;
    to?: number;
}

export interface AnimDef {
    cycleSec?: number;          // 计时动作一个循环的秒数；计次动作用动作库的 secPerRep
    keys: [number, Pose][];     // [循环内的进度 0..1, 姿态]，第一个必须是 0；最后自动接回第一个
    anchor?: Anchor;            // 哪个部位在水平面上保持不动（默认骨盆不动）
    hang?: boolean;             // 悬垂：手固定在单杠高度，不贴地
    plant?: { part: 'footL' | 'footR'; y: number };  // 这只脚踩在高处（凳面高度 y），身体其余部分不低于地面
    repsPerCycle?: number;      // 一个循环包含几次（左右交替的动作 = 2），默认 1
    // 撑地不动的部位。不写时：anchor 为 feet 且不跳跃 → 双脚整个循环不动；anchor 为 hands → 双手不动
    contacts?: Contact[];
    ropeTurns?: number;         // 跳绳：一个循环里绳子转几圈，默认 1
    props?: BodyProp[];
    scene?: SceneProp[];
    linear?: boolean;           // 关键帧之间匀速（跑、跳）；默认缓入缓出
}
