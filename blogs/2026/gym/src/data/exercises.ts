import type { Exercise } from '../types';

type ExerciseInput = Pick<Exercise, 'id' | 'name' | 'phases' | 'category' | 'muscles' | 'sweat' | 'intensity' | 'cues'> &
    Partial<Exercise>;

// 默认：徒手、室内外均可、不需要鞋、原地、安静、计次、每次 3 秒
function ex(input: ExerciseInput): Exercise {
    return {
        equipment: [],
        locations: ['indoor', 'outdoor'],
        needsShoes: false,
        needsSpace: false,
        noisy: false,
        stresses: [],
        mode: 'reps',
        secPerRep: 3,
        ...input,
    };
}

export const EXERCISES: Exercise[] = [
    // ---------- 热身 / 激活 ----------
    ex({
        id: 'arm_circles', name: '手臂绕环', phases: ['warmup'], category: 'mobility',
        muscles: ['shoulders'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30,
        cues: ['手臂伸直，由小圈到大圈', '前后方向各做一半时间'],
    }),
    ex({
        id: 'hip_circles', name: '髋关节绕环', phases: ['warmup'], category: 'mobility',
        muscles: ['glutes'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30,
        cues: ['双手叉腰，髋部画大圈', '顺时针、逆时针各做一半时间'],
    }),
    ex({
        id: 'torso_twist', name: '站姿转体', phases: ['warmup'], category: 'mobility',
        muscles: ['core'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30,
        cues: ['双脚与肩同宽，骨盆朝前', '上身左右转动，手臂自然甩动'],
    }),
    ex({
        id: 'leg_swings', name: '腿部前后摆动', phases: ['warmup'], category: 'mobility',
        muscles: ['legs', 'glutes'], sweat: 1, intensity: 1, defaultReps: 10, secPerRep: 1.5, perSide: true,
        cues: ['可扶墙保持平衡', '摆动幅度逐渐加大，上身保持直立'],
    }),
    ex({
        id: 'march_in_place', name: '原地踏步抬膝', phases: ['warmup'], category: 'cardio',
        muscles: ['legs', 'core'], sweat: 2, intensity: 1, mode: 'time', defaultSeconds: 45,
        cues: ['膝盖抬到髋部高度', '手臂配合摆动，落地轻柔'],
    }),
    ex({
        id: 'inchworm', name: '尺蠖爬', phases: ['warmup'], category: 'mobility',
        muscles: ['full_body'], sweat: 2, intensity: 2, defaultReps: 6, secPerRep: 6, stresses: ['wrist', 'back'],
        cues: ['俯身双手撑地，手向前爬到平板位', '再用脚向手的方向小步走回'],
    }),
    ex({
        id: 'cat_cow', name: '猫牛式', equipment: ['mat'], phases: ['warmup', 'cooldown'], category: 'mobility',
        muscles: ['back', 'core'], sweat: 1, intensity: 1, defaultReps: 10, secPerRep: 4, stresses: ['wrist'],
        cues: ['四点跪姿，吸气塌腰抬头', '呼气拱背低头，动作随呼吸缓慢进行'],
    }),
    ex({
        id: 'worlds_greatest_stretch', name: '世界最伟大拉伸', phases: ['warmup'], category: 'mobility',
        muscles: ['legs', 'glutes', 'back'], sweat: 1, intensity: 2, defaultReps: 5, secPerRep: 6, perSide: true,
        stresses: ['wrist'],
        cues: ['弓步，同侧手肘向下沉', '再向上打开转体，眼睛看向上方手'],
    }),
    ex({
        id: 'bw_good_morning', name: '徒手早安式', phases: ['warmup'], category: 'mobility',
        muscles: ['glutes', 'legs'], sweat: 1, intensity: 1, defaultReps: 12, stresses: ['back'],
        cues: ['双手抱头，膝盖微屈', '臀部向后推，背部保持平直'],
    }),
    ex({
        id: 'jumping_jacks', name: '开合跳', phases: ['warmup', 'main'], category: 'cardio',
        muscles: ['full_body'], sweat: 3, intensity: 3, mode: 'time', defaultSeconds: 30,
        needsShoes: true, noisy: true, stresses: ['knee'],
        cues: ['前脚掌落地，膝盖微屈缓冲', '手臂伸直上举'],
    }),
    ex({
        id: 'jump_rope_easy', name: '慢速跳绳', equipment: ['jump_rope'], phases: ['warmup'], category: 'cardio',
        muscles: ['legs'], sweat: 2, intensity: 2, mode: 'time', defaultSeconds: 60,
        needsShoes: true, noisy: true, needsSpace: true, stresses: ['knee'],
        cues: ['节奏放慢，跳离地面一点即可', '手腕转绳，大臂贴近身体'],
    }),
    ex({
        id: 'light_jog', name: '慢跑', locations: ['outdoor'], phases: ['warmup'], category: 'cardio',
        muscles: ['legs'], sweat: 2, intensity: 2, mode: 'time', defaultSeconds: 120,
        needsShoes: true, needsSpace: true, stresses: ['knee'],
        cues: ['能边跑边说话的速度', '步频轻快，脚落在身体下方'],
    }),
    ex({
        id: 'band_pull_apart', name: '弹力带拉开', equipment: ['band'], phases: ['warmup', 'main'], category: 'strength',
        muscles: ['back', 'shoulders'], sweat: 1, intensity: 1, defaultReps: 15, secPerRep: 2,
        cues: ['手臂伸直与肩同高', '向两侧拉开，肩胛骨向中间夹'],
    }),
    ex({
        id: 'kb_halo', name: '壶铃绕头', equipment: ['kettlebell'], phases: ['warmup', 'main'], category: 'mobility',
        muscles: ['shoulders', 'core'], sweat: 1, intensity: 2, defaultReps: 8, stresses: ['shoulder'],
        cues: ['倒握壶铃把手，绕头部画圈', '核心收紧，身体不要跟着晃'],
    }),

    // ---------- 主训练：徒手力量 ----------
    ex({
        id: 'squat', name: '徒手深蹲', phases: ['main'], category: 'strength',
        muscles: ['legs', 'glutes'], sweat: 2, intensity: 2, defaultReps: 15, stresses: ['knee'],
        cues: ['双脚与肩同宽，脚尖略外八', '臀部向后坐，膝盖与脚尖同向'],
    }),
    ex({
        id: 'tempo_squat', name: '慢速深蹲（3秒下蹲）', phases: ['main'], category: 'strength',
        muscles: ['legs', 'glutes'], sweat: 2, intensity: 3, defaultReps: 10, secPerRep: 5, stresses: ['knee'],
        cues: ['下蹲数 3 秒，底部停 1 秒', '站起正常速度，全程控制'],
    }),
    ex({
        id: 'lunge', name: '箭步蹲', phases: ['main'], category: 'strength',
        muscles: ['legs', 'glutes'], sweat: 2, intensity: 3, defaultReps: 10, perSide: true, stresses: ['knee'],
        cues: ['前后腿都约成 90 度', '前膝不内扣，上身保持直立'],
    }),
    ex({
        id: 'glute_bridge', name: '臀桥', equipment: ['mat'], phases: ['main'], category: 'strength',
        muscles: ['glutes'], sweat: 1, intensity: 2, defaultReps: 15, secPerRep: 3,
        cues: ['仰卧屈膝，脚跟发力抬起臀部', '顶端夹紧臀部停 1 秒'],
    }),
    ex({
        id: 'calf_raise', name: '提踵', phases: ['main'], category: 'strength',
        muscles: ['legs'], sweat: 1, intensity: 1, defaultReps: 20, secPerRep: 2,
        cues: ['可扶墙，踮脚到最高点', '缓慢下放'],
    }),
    ex({
        id: 'wall_sit', name: '靠墙静蹲', locations: ['indoor'], phases: ['main'], category: 'strength',
        muscles: ['legs'], sweat: 2, intensity: 3, mode: 'time', defaultSeconds: 40, stresses: ['knee'],
        cues: ['背贴墙，大腿接近与地面平行', '膝盖不超过脚尖太多'],
    }),
    ex({
        id: 'pushup', name: '俯卧撑', phases: ['main'], category: 'strength',
        muscles: ['chest', 'arms'], sweat: 2, intensity: 3, defaultReps: 10, stresses: ['wrist', 'shoulder'],
        cues: ['身体成一条直线', '手肘与身体约 45 度，胸部接近地面'],
    }),
    ex({
        id: 'knee_pushup', name: '跪姿俯卧撑', equipment: ['mat'], phases: ['main'], category: 'strength',
        muscles: ['chest', 'arms'], sweat: 1, intensity: 2, defaultReps: 12, stresses: ['wrist'],
        cues: ['膝盖着地，膝到头成一条直线', '下放时控制速度'],
    }),
    ex({
        id: 'incline_pushup', name: '上斜俯卧撑', equipment: ['chair'], locations: ['indoor'], phases: ['main'],
        category: 'strength', muscles: ['chest', 'arms'], sweat: 1, intensity: 2, defaultReps: 12, stresses: ['wrist'],
        cues: ['双手扶稳固的椅子或桌沿', '身体保持直线'],
    }),
    ex({
        id: 'pike_pushup', name: '折刀俯卧撑', phases: ['main'], category: 'strength',
        muscles: ['shoulders', 'arms'], sweat: 2, intensity: 4, defaultReps: 8, stresses: ['wrist', 'shoulder'],
        cues: ['臀部抬高成倒 V', '头部向两手之间下放'],
    }),
    ex({
        id: 'chair_dip', name: '椅子臂屈伸', equipment: ['chair'], phases: ['main'], category: 'strength',
        muscles: ['arms'], sweat: 2, intensity: 3, defaultReps: 10, stresses: ['shoulder', 'wrist'],
        cues: ['椅子要稳固，手指朝前', '肘部向后弯，不要外张'],
    }),
    ex({
        id: 'step_up', name: '登台阶', equipment: ['chair'], phases: ['main'], category: 'strength',
        muscles: ['legs', 'glutes'], sweat: 3, intensity: 3, defaultReps: 10, secPerRep: 3, perSide: true, stresses: ['knee'],
        cues: ['整只脚踩上台面', '用上方腿发力站起，不要用后脚蹬'],
    }),
    ex({
        id: 'bulgarian_split_squat', name: '保加利亚分腿蹲', equipment: ['chair'], phases: ['main'], category: 'strength',
        muscles: ['legs', 'glutes'], sweat: 2, intensity: 4, defaultReps: 8, perSide: true, stresses: ['knee'],
        cues: ['后脚搭在椅子上', '前腿发力，身体垂直下降'],
    }),
    ex({
        id: 'superman', name: '超人式', equipment: ['mat'], phases: ['main'], category: 'strength',
        muscles: ['back'], sweat: 1, intensity: 2, defaultReps: 12, secPerRep: 4,
        cues: ['俯卧，同时抬起手臂和腿', '顶端停 2 秒，颈部保持中立'],
    }),
    ex({
        id: 'pullup', name: '引体向上', equipment: ['pullup_bar'], phases: ['main'], category: 'strength',
        muscles: ['back', 'arms'], sweat: 2, intensity: 4, defaultReps: 6, secPerRep: 4, stresses: ['shoulder'],
        cues: ['正握略宽于肩', '下巴过杆，下放到手臂伸直'],
    }),
    ex({
        id: 'hanging_knee_raise', name: '悬垂举腿', equipment: ['pullup_bar'], phases: ['main'], category: 'core',
        muscles: ['core'], sweat: 2, intensity: 3, defaultReps: 10, stresses: ['shoulder'],
        cues: ['悬挂在杆上，收腹把膝盖抬向胸口', '避免身体前后晃动'],
    }),

    // ---------- 主训练：核心 ----------
    ex({
        id: 'plank', name: '平板支撑', equipment: ['mat'], phases: ['main'], category: 'core',
        muscles: ['core'], sweat: 1, intensity: 2, mode: 'time', defaultSeconds: 30,
        cues: ['肘在肩正下方', '收腹夹臀，身体成一条直线'],
    }),
    ex({
        id: 'side_plank', name: '侧平板支撑', equipment: ['mat'], phases: ['main'], category: 'core',
        muscles: ['core'], sweat: 1, intensity: 2, mode: 'time', defaultSeconds: 20, perSide: true,
        stresses: ['shoulder'],
        cues: ['肘在肩正下方', '髋部抬起，不要下塌'],
    }),
    ex({
        id: 'dead_bug', name: '死虫式', equipment: ['mat'], phases: ['main'], category: 'core',
        muscles: ['core'], sweat: 1, intensity: 1, defaultReps: 10, perSide: true,
        cues: ['仰卧，腰背贴地', '对侧手脚缓慢伸展'],
    }),
    ex({
        id: 'bird_dog', name: '鸟狗式', equipment: ['mat'], phases: ['main'], category: 'core',
        muscles: ['core', 'back'], sweat: 1, intensity: 1, defaultReps: 10, perSide: true,
        cues: ['四点跪姿，对侧手脚同时伸直', '骨盆保持水平'],
    }),
    ex({
        id: 'crunch', name: '卷腹', equipment: ['mat'], phases: ['main'], category: 'core',
        muscles: ['core'], sweat: 1, intensity: 2, defaultReps: 15, secPerRep: 2,
        cues: ['肩胛骨离地即可', '不要用手拉脖子'],
    }),
    ex({
        id: 'bicycle_crunch', name: '自行车卷腹', equipment: ['mat'], phases: ['main'], category: 'core',
        muscles: ['core'], sweat: 2, intensity: 3, defaultReps: 20, secPerRep: 1.5,
        cues: ['手肘向对侧膝盖靠近', '动作慢而有控制'],
    }),
    ex({
        id: 'standing_crunch', name: '站姿交替提膝卷腹', phases: ['main'], category: 'core',
        muscles: ['core'], sweat: 1, intensity: 2, defaultReps: 20, secPerRep: 1.5,
        cues: ['手抱头，膝盖抬向对侧手肘', '收腹完成动作'],
    }),

    // ---------- 主训练：有氧 / 高强度 ----------
    ex({
        id: 'mountain_climber', name: '登山者', phases: ['main'], category: 'cardio',
        muscles: ['core', 'full_body'], sweat: 4, intensity: 4, mode: 'time', defaultSeconds: 30,
        stresses: ['wrist', 'shoulder'],
        cues: ['平板支撑位，交替快速提膝', '臀部不要抬太高'],
    }),
    ex({
        id: 'burpee', name: '波比跳', phases: ['main'], category: 'cardio',
        muscles: ['full_body'], sweat: 5, intensity: 5, mode: 'time', defaultSeconds: 30,
        needsShoes: true, noisy: true, stresses: ['knee', 'wrist', 'back'],
        cues: ['下蹲、后跳成平板、收腿、跳起', '落地轻柔，量力而行'],
    }),
    ex({
        id: 'high_knees', name: '高抬腿跑', phases: ['main'], category: 'cardio',
        muscles: ['legs', 'core'], sweat: 4, intensity: 4, mode: 'time', defaultSeconds: 30,
        needsShoes: true, noisy: true, stresses: ['knee'],
        cues: ['原地快速抬膝', '前脚掌着地，手臂积极摆动'],
    }),
    ex({
        id: 'squat_jump', name: '深蹲跳', phases: ['main'], category: 'cardio',
        muscles: ['legs', 'glutes'], sweat: 4, intensity: 4, defaultReps: 10, secPerRep: 2,
        needsShoes: true, noisy: true, stresses: ['knee'],
        cues: ['下蹲后爆发跳起', '落地屈膝缓冲，直接进入下一次'],
    }),
    ex({
        id: 'skater', name: '滑冰跳', phases: ['main'], category: 'cardio',
        muscles: ['legs', 'glutes'], sweat: 4, intensity: 4, mode: 'time', defaultSeconds: 30,
        needsShoes: true, noisy: true, needsSpace: true, stresses: ['knee'],
        cues: ['向侧面跳，单脚落地', '后腿向斜后方交叉'],
    }),
    ex({
        id: 'jump_rope', name: '跳绳', equipment: ['jump_rope'], phases: ['main'], category: 'cardio',
        muscles: ['legs', 'full_body'], sweat: 4, intensity: 4, mode: 'time', defaultSeconds: 60,
        needsShoes: true, noisy: true, needsSpace: true, stresses: ['knee'],
        cues: ['前脚掌着地，膝盖微屈', '保持稳定节奏'],
    }),
    ex({
        id: 'sprint', name: '冲刺跑', locations: ['outdoor'], phases: ['main'], category: 'cardio',
        muscles: ['legs'], sweat: 5, intensity: 5, mode: 'time', defaultSeconds: 20,
        needsShoes: true, needsSpace: true, stresses: ['knee'],
        cues: ['充分热身后再做', '全力跑后走回起点恢复'],
    }),
    ex({
        id: 'shadow_boxing', name: '空击拳', phases: ['main'], category: 'cardio',
        muscles: ['shoulders', 'core'], sweat: 3, intensity: 3, mode: 'time', defaultSeconds: 40,
        cues: ['两脚前后站，膝盖微屈', '出拳时转腰送肩，快速收回'],
    }),

    // ---------- 主训练：哑铃 ----------
    ex({
        id: 'db_goblet_squat', name: '哑铃高脚杯深蹲', equipment: ['dumbbell'], phases: ['main'], category: 'strength',
        muscles: ['legs', 'glutes'], sweat: 3, intensity: 3, defaultReps: 12, stresses: ['knee'],
        cues: ['双手托住哑铃一端于胸前', '下蹲时手肘在膝盖内侧'],
    }),
    ex({
        id: 'db_rdl', name: '哑铃罗马尼亚硬拉', equipment: ['dumbbell'], phases: ['main'], category: 'strength',
        muscles: ['glutes', 'legs', 'back'], sweat: 2, intensity: 3, defaultReps: 12, stresses: ['back'],
        cues: ['膝盖微屈，臀部向后推', '哑铃贴腿下放，背部始终平直'],
    }),
    ex({
        id: 'db_lunge', name: '哑铃箭步蹲', equipment: ['dumbbell'], phases: ['main'], category: 'strength',
        muscles: ['legs', 'glutes'], sweat: 3, intensity: 3, defaultReps: 10, perSide: true, stresses: ['knee'],
        cues: ['双手持哑铃于体侧', '前膝不内扣'],
    }),
    ex({
        id: 'db_row', name: '哑铃单臂划船', equipment: ['dumbbell'], phases: ['main'], category: 'strength',
        muscles: ['back', 'arms'], sweat: 2, intensity: 3, defaultReps: 10, perSide: true,
        cues: ['一手撑在膝盖或椅子上，背部平直', '手肘贴身向后上方拉'],
    }),
    ex({
        id: 'db_floor_press', name: '哑铃地板卧推', equipment: ['dumbbell'], locations: ['indoor'], phases: ['main'],
        category: 'strength', muscles: ['chest', 'arms'], sweat: 2, intensity: 3, defaultReps: 10, stresses: ['shoulder'],
        cues: ['仰卧屈膝，上臂落地即停', '向上推起至手臂伸直'],
    }),
    ex({
        id: 'db_shoulder_press', name: '哑铃推举', equipment: ['dumbbell'], phases: ['main'], category: 'strength',
        muscles: ['shoulders', 'arms'], sweat: 2, intensity: 3, defaultReps: 10, stresses: ['shoulder'],
        cues: ['哑铃从肩部向上推', '核心收紧，不要塌腰'],
    }),
    ex({
        id: 'db_lateral_raise', name: '哑铃侧平举', equipment: ['dumbbell'], phases: ['main'], category: 'strength',
        muscles: ['shoulders'], sweat: 1, intensity: 2, defaultReps: 12, stresses: ['shoulder'],
        cues: ['手肘微屈，抬到与肩同高', '慢速下放'],
    }),
    ex({
        id: 'db_curl', name: '哑铃弯举', equipment: ['dumbbell'], phases: ['main'], category: 'strength',
        muscles: ['arms'], sweat: 1, intensity: 2, defaultReps: 12,
        cues: ['上臂贴紧身体', '顶端挤压，慢速下放'],
    }),
    ex({
        id: 'db_thruster', name: '哑铃深蹲推举', equipment: ['dumbbell'], phases: ['main'], category: 'cardio',
        muscles: ['full_body'], sweat: 4, intensity: 4, defaultReps: 10, stresses: ['knee', 'shoulder'],
        cues: ['哑铃在肩，下蹲', '站起时借力把哑铃推过头顶'],
    }),

    // ---------- 主训练：杠铃 ----------
    ex({
        id: 'bb_squat', name: '杠铃深蹲', equipment: ['barbell'], locations: ['indoor'], phases: ['main'],
        category: 'strength', muscles: ['legs', 'glutes'], sweat: 3, intensity: 4, defaultReps: 8,
        stresses: ['knee', 'back'],
        cues: ['杠铃放在斜方肌上', '下蹲至大腿与地面平行，背部挺直'],
    }),
    ex({
        id: 'bb_deadlift', name: '杠铃硬拉', equipment: ['barbell'], phases: ['main'], category: 'strength',
        muscles: ['back', 'glutes', 'legs'], sweat: 3, intensity: 5, defaultReps: 6, secPerRep: 4, stresses: ['back'],
        cues: ['杠铃贴近小腿，背部平直', '腿和髋同时发力站起'],
    }),
    ex({
        id: 'bb_bent_row', name: '杠铃俯身划船', equipment: ['barbell'], phases: ['main'], category: 'strength',
        muscles: ['back', 'arms'], sweat: 2, intensity: 4, defaultReps: 10, stresses: ['back'],
        cues: ['俯身约 45 度，背部平直', '把杠铃拉向腹部'],
    }),
    ex({
        id: 'bb_overhead_press', name: '杠铃站姿推举', equipment: ['barbell'], phases: ['main'], category: 'strength',
        muscles: ['shoulders', 'arms'], sweat: 2, intensity: 4, defaultReps: 8, stresses: ['shoulder', 'back'],
        cues: ['杠铃从锁骨位置推起', '收紧臀部和核心，不要后仰'],
    }),

    // ---------- 主训练：壶铃 ----------
    ex({
        id: 'kb_swing', name: '壶铃摆荡', equipment: ['kettlebell'], phases: ['main'], category: 'cardio',
        muscles: ['glutes', 'back', 'full_body'], sweat: 4, intensity: 4, defaultReps: 15, secPerRep: 2, needsSpace: true,
        stresses: ['back'],
        cues: ['髋部发力向前顶，不是用手臂举', '壶铃摆到胸口高度'],
    }),
    ex({
        id: 'kb_goblet_squat', name: '壶铃高脚杯深蹲', equipment: ['kettlebell'], phases: ['main'], category: 'strength',
        muscles: ['legs', 'glutes'], sweat: 3, intensity: 3, defaultReps: 12, stresses: ['knee'],
        cues: ['双手握壶铃把手于胸前', '下蹲时保持上身直立'],
    }),

    // ---------- 主训练：弹力带 ----------
    ex({
        id: 'band_row', name: '弹力带划船', equipment: ['band'], phases: ['main'], category: 'strength',
        muscles: ['back', 'arms'], sweat: 1, intensity: 2, defaultReps: 15,
        cues: ['弹力带固定或踩在脚下', '肩胛骨后缩，手肘向后拉'],
    }),
    ex({
        id: 'band_chest_press', name: '弹力带推胸', equipment: ['band'], phases: ['main'], category: 'strength',
        muscles: ['chest', 'arms'], sweat: 1, intensity: 2, defaultReps: 15, stresses: ['shoulder'],
        cues: ['弹力带绕过背后', '向前推至手臂伸直'],
    }),
    ex({
        id: 'band_squat', name: '弹力带深蹲', equipment: ['band'], phases: ['main'], category: 'strength',
        muscles: ['legs', 'glutes'], sweat: 2, intensity: 2, defaultReps: 15, stresses: ['knee'],
        cues: ['踩住弹力带，两端拉至肩部', '下蹲站起，膝盖向外撑'],
    }),
    ex({
        id: 'band_lateral_walk', name: '弹力带侧向走', equipment: ['band'], phases: ['main'], category: 'strength',
        muscles: ['glutes'], sweat: 2, intensity: 2, defaultReps: 10, secPerRep: 2, perSide: true,
        cues: ['弹力带套在膝上或脚踝', '半蹲姿态向侧面小步走'],
    }),
    ex({
        id: 'band_face_pull', name: '弹力带面拉', equipment: ['band'], phases: ['main'], category: 'strength',
        muscles: ['shoulders', 'back'], sweat: 1, intensity: 1, defaultReps: 15,
        cues: ['弹力带固定在面部高度', '拉向额头，手肘高于手腕'],
    }),

    // ---------- 柔韧 / 放松 ----------
    ex({
        id: 'standing_quad_stretch', name: '站姿股四头肌拉伸', phases: ['main', 'cooldown'], category: 'stretch',
        muscles: ['legs'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, perSide: true,
        cues: ['可扶墙，手抓脚踝拉向臀部', '两膝并拢，骨盆前推'],
    }),
    ex({
        id: 'standing_hamstring_stretch', name: '站姿腘绳肌拉伸', phases: ['main', 'cooldown'], category: 'stretch',
        muscles: ['legs'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, perSide: true,
        cues: ['一脚前伸脚跟着地', '背部平直，从髋部向前俯身'],
    }),
    ex({
        id: 'calf_stretch', name: '小腿拉伸', phases: ['cooldown'], category: 'stretch',
        muscles: ['legs'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, perSide: true,
        cues: ['双手扶墙，后腿伸直脚跟踩地', '身体前倾感受小腿拉伸'],
    }),
    ex({
        id: 'figure_four_stretch', name: '站姿 4 字拉伸', phases: ['main', 'cooldown'], category: 'stretch',
        muscles: ['glutes'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, perSide: true,
        cues: ['一脚踝搭在另一侧膝上', '像坐椅子一样下蹲，可扶墙'],
    }),
    ex({
        id: 'chest_wall_stretch', name: '扶墙胸部拉伸', phases: ['main', 'cooldown'], category: 'stretch',
        muscles: ['chest', 'shoulders'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, perSide: true,
        cues: ['手臂贴墙或树干', '身体向另一侧转开'],
    }),
    ex({
        id: 'cross_body_shoulder_stretch', name: '肩部交叉拉伸', phases: ['cooldown'], category: 'stretch',
        muscles: ['shoulders'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, perSide: true,
        cues: ['一臂横过胸前', '另一手压住上臂靠近身体'],
    }),
    ex({
        id: 'triceps_stretch', name: '肱三头肌拉伸', phases: ['cooldown'], category: 'stretch',
        muscles: ['arms'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, perSide: true,
        cues: ['手臂上举屈肘，手摸后背', '另一手轻压手肘'],
    }),
    ex({
        id: 'standing_side_bend', name: '站姿侧屈拉伸', phases: ['main', 'cooldown'], category: 'stretch',
        muscles: ['core', 'back'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, perSide: true,
        cues: ['手臂上举，向一侧弯曲', '髋部保持不动'],
    }),
    ex({
        id: 'deep_breathing', name: '腹式呼吸', phases: ['cooldown'], category: 'stretch',
        muscles: ['core'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 60,
        cues: ['吸气 4 秒腹部鼓起', '呼气 6 秒，全身放松'],
    }),
    ex({
        id: 'child_pose', name: '婴儿式', equipment: ['mat'], phases: ['main', 'cooldown'], category: 'stretch',
        muscles: ['back'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 45, stresses: ['knee'],
        cues: ['跪坐，上身前趴', '手臂前伸，额头触地，深呼吸'],
    }),
    ex({
        id: 'downward_dog', name: '下犬式', equipment: ['mat'], phases: ['warmup', 'main', 'cooldown'],
        category: 'stretch', muscles: ['legs', 'shoulders'], sweat: 1, intensity: 2, mode: 'time',
        defaultSeconds: 30, stresses: ['wrist', 'shoulder'],
        cues: ['臀部向上推成倒 V', '脚跟尽量向下踩'],
    }),
    ex({
        id: 'cobra', name: '眼镜蛇式', equipment: ['mat'], phases: ['main', 'cooldown'], category: 'stretch',
        muscles: ['core', 'back'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, stresses: ['back'],
        cues: ['俯卧，双手撑起上身', '耻骨贴地，肩膀下沉'],
    }),
    ex({
        id: 'pigeon_pose', name: '鸽子式', equipment: ['mat'], phases: ['main', 'cooldown'], category: 'stretch',
        muscles: ['glutes'], sweat: 1, intensity: 2, mode: 'time', defaultSeconds: 45, perSide: true,
        stresses: ['knee'],
        cues: ['前腿屈膝横放，后腿向后伸直', '上身慢慢前趴'],
    }),
    ex({
        id: 'seated_forward_fold', name: '坐姿体前屈', equipment: ['mat'], phases: ['main', 'cooldown'],
        category: 'stretch', muscles: ['legs', 'back'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 45,
        cues: ['双腿伸直坐地', '从髋部向前折叠，不要强行压背'],
    }),
    ex({
        id: 'supine_twist', name: '仰卧扭转', equipment: ['mat'], phases: ['main', 'cooldown'], category: 'stretch',
        muscles: ['back', 'core'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, perSide: true,
        cues: ['仰卧，一侧膝盖倒向另一侧', '双肩贴地，眼看反方向'],
    }),
    ex({
        id: 'kneeling_hip_flexor_stretch', name: '跪姿髋屈肌拉伸', equipment: ['mat'], phases: ['main', 'cooldown'],
        category: 'stretch', muscles: ['legs', 'glutes'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30,
        perSide: true, stresses: ['knee'],
        cues: ['单膝跪地成弓步', '夹紧后侧臀部，骨盆向前推'],
    }),
    ex({
        id: 'warrior_two', name: '战士二式', phases: ['warmup', 'main'], category: 'mobility',
        muscles: ['legs', 'shoulders'], sweat: 1, intensity: 2, mode: 'time', defaultSeconds: 30, perSide: true,
        stresses: ['knee'],
        cues: ['两脚大跨步，前膝弯曲对准脚尖', '双臂平展，目视前手'],
    }),
    ex({
        id: 'deep_squat_hold', name: '深蹲静态停留', phases: ['main', 'cooldown'], category: 'mobility',
        muscles: ['legs', 'glutes'], sweat: 1, intensity: 1, mode: 'time', defaultSeconds: 30, stresses: ['knee'],
        cues: ['尽量蹲到底，脚跟着地', '双手合十，手肘撑开膝盖'],
    }),
];

export const EXERCISE_BY_ID: Record<string, Exercise> = Object.fromEntries(EXERCISES.map((e) => [e.id, e]));
