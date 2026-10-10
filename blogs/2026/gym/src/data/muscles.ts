// 每个动作锻炼（或拉伸）到的具体肌肉，用于人体肌肉图。
// 比 Exercise.muscles 的大类更细；新动作加在 exercises.ts 后也要在这里补上（有测试检查）

export type MuscleDetail =
    | 'chest'
    | 'delts'
    | 'traps'
    | 'lats'
    | 'lower_back'
    | 'biceps'
    | 'triceps'
    | 'forearms'
    | 'abs'
    | 'obliques'
    | 'hip_flexors'
    | 'glutes'
    | 'quads'
    | 'hamstrings'
    | 'adductors'
    | 'calves';

export const MUSCLE_DETAIL_LABEL: Record<MuscleDetail, string> = {
    chest: '胸肌',
    delts: '三角肌（肩）',
    traps: '斜方肌',
    lats: '背阔肌',
    lower_back: '下背（竖脊肌）',
    biceps: '肱二头肌',
    triceps: '肱三头肌',
    forearms: '前臂',
    abs: '腹直肌',
    obliques: '腹斜肌',
    hip_flexors: '髋屈肌',
    glutes: '臀肌',
    quads: '股四头肌',
    hamstrings: '腘绳肌',
    adductors: '大腿内侧（内收肌）',
    calves: '小腿',
};

export interface ExerciseMuscles {
    primary: MuscleDetail[];
    secondary: MuscleDetail[];
}

function m(primary: MuscleDetail[], secondary: MuscleDetail[] = []): ExerciseMuscles {
    return { primary, secondary };
}

export const EXERCISE_MUSCLES: Record<string, ExerciseMuscles> = {
    // ---------- 热身 / 激活 ----------
    arm_circles: m(['delts'], ['traps']),
    hip_circles: m(['glutes', 'hip_flexors'], ['obliques', 'lower_back']),
    torso_twist: m(['obliques'], ['abs', 'lower_back']),
    leg_swings: m(['hip_flexors', 'hamstrings'], ['glutes']),
    march_in_place: m(['hip_flexors', 'quads'], ['calves', 'abs']),
    inchworm: m(['hamstrings', 'abs', 'delts'], ['triceps', 'chest']),
    cat_cow: m(['lower_back', 'abs'], ['traps']),
    worlds_greatest_stretch: m(['hip_flexors', 'hamstrings', 'glutes'], ['obliques', 'lats']),
    bw_good_morning: m(['hamstrings', 'glutes'], ['lower_back']),
    jumping_jacks: m(['calves', 'delts', 'quads'], ['adductors', 'glutes']),
    jump_rope_easy: m(['calves'], ['quads', 'forearms']),
    light_jog: m(['quads', 'calves'], ['hamstrings', 'glutes']),
    band_pull_apart: m(['delts', 'traps'], ['lats']),
    kb_halo: m(['delts'], ['abs', 'obliques', 'traps']),

    // ---------- 主训练 ----------
    squat: m(['quads', 'glutes'], ['hamstrings', 'adductors', 'abs']),
    tempo_squat: m(['quads', 'glutes'], ['hamstrings', 'adductors', 'abs']),
    lunge: m(['quads', 'glutes'], ['hamstrings', 'adductors', 'calves']),
    glute_bridge: m(['glutes'], ['hamstrings', 'lower_back']),
    calf_raise: m(['calves']),
    wall_sit: m(['quads'], ['glutes', 'adductors']),
    pushup: m(['chest', 'triceps'], ['delts', 'abs']),
    knee_pushup: m(['chest', 'triceps'], ['delts', 'abs']),
    incline_pushup: m(['chest', 'triceps'], ['delts', 'abs']),
    pike_pushup: m(['delts', 'triceps'], ['chest', 'traps']),
    chair_dip: m(['triceps'], ['chest', 'delts']),
    step_up: m(['quads', 'glutes'], ['hamstrings', 'calves']),
    bulgarian_split_squat: m(['quads', 'glutes'], ['hamstrings', 'adductors']),
    superman: m(['lower_back', 'glutes'], ['hamstrings', 'traps']),
    pullup: m(['lats', 'biceps'], ['forearms', 'traps', 'delts']),
    hanging_knee_raise: m(['abs', 'hip_flexors'], ['forearms', 'obliques']),
    plank: m(['abs'], ['obliques', 'delts', 'glutes']),
    side_plank: m(['obliques'], ['abs', 'delts', 'glutes']),
    dead_bug: m(['abs'], ['hip_flexors', 'obliques']),
    bird_dog: m(['lower_back', 'abs'], ['glutes', 'delts']),
    crunch: m(['abs'], ['obliques']),
    bicycle_crunch: m(['obliques', 'abs'], ['hip_flexors']),
    standing_crunch: m(['abs', 'obliques'], ['hip_flexors']),
    mountain_climber: m(['abs', 'hip_flexors'], ['delts', 'quads', 'chest']),
    burpee: m(['quads', 'chest'], ['glutes', 'delts', 'triceps', 'abs', 'calves']),
    high_knees: m(['hip_flexors', 'quads'], ['calves', 'abs']),
    squat_jump: m(['quads', 'glutes'], ['calves', 'hamstrings']),
    skater: m(['glutes', 'quads'], ['adductors', 'calves']),
    jump_rope: m(['calves'], ['quads', 'delts', 'forearms']),
    sprint: m(['quads', 'hamstrings', 'glutes'], ['calves', 'hip_flexors']),
    shadow_boxing: m(['delts'], ['obliques', 'triceps', 'abs']),
    db_goblet_squat: m(['quads', 'glutes'], ['adductors', 'hamstrings', 'abs']),
    db_rdl: m(['hamstrings', 'glutes'], ['lower_back', 'forearms']),
    db_lunge: m(['quads', 'glutes'], ['hamstrings', 'adductors', 'forearms']),
    db_row: m(['lats', 'biceps'], ['traps', 'delts', 'forearms']),
    db_floor_press: m(['chest', 'triceps'], ['delts']),
    db_shoulder_press: m(['delts', 'triceps'], ['traps']),
    db_lateral_raise: m(['delts'], ['traps']),
    db_curl: m(['biceps'], ['forearms']),
    db_thruster: m(['quads', 'glutes', 'delts'], ['triceps', 'abs']),
    bb_squat: m(['quads', 'glutes'], ['hamstrings', 'adductors', 'lower_back']),
    bb_deadlift: m(['hamstrings', 'glutes', 'lower_back'], ['traps', 'forearms', 'quads', 'lats']),
    bb_bent_row: m(['lats', 'traps'], ['biceps', 'lower_back', 'delts']),
    bb_overhead_press: m(['delts', 'triceps'], ['traps', 'abs']),
    kb_swing: m(['glutes', 'hamstrings'], ['lower_back', 'delts', 'abs']),
    kb_goblet_squat: m(['quads', 'glutes'], ['adductors', 'abs']),
    band_row: m(['lats', 'biceps'], ['traps', 'delts']),
    band_chest_press: m(['chest', 'triceps'], ['delts']),
    band_squat: m(['quads', 'glutes'], ['hamstrings']),
    band_lateral_walk: m(['glutes'], ['quads']),
    band_face_pull: m(['delts', 'traps'], ['biceps']),

    // ---------- 放松 / 拉伸 ----------
    standing_quad_stretch: m(['quads', 'hip_flexors']),
    standing_hamstring_stretch: m(['hamstrings'], ['calves']),
    calf_stretch: m(['calves']),
    figure_four_stretch: m(['glutes']),
    chest_wall_stretch: m(['chest'], ['delts', 'biceps']),
    cross_body_shoulder_stretch: m(['delts'], ['traps']),
    triceps_stretch: m(['triceps'], ['lats']),
    standing_side_bend: m(['obliques', 'lats']),
    deep_breathing: m(['abs']),
    child_pose: m(['lower_back', 'lats'], ['glutes']),
    downward_dog: m(['hamstrings', 'calves', 'delts'], ['lats']),
    cobra: m(['abs', 'lower_back'], ['hip_flexors']),
    pigeon_pose: m(['glutes', 'hip_flexors']),
    seated_forward_fold: m(['hamstrings', 'lower_back'], ['calves']),
    supine_twist: m(['obliques', 'lower_back'], ['glutes']),
    kneeling_hip_flexor_stretch: m(['hip_flexors', 'quads']),
    warrior_two: m(['quads', 'delts'], ['adductors', 'glutes']),
    deep_squat_hold: m(['adductors', 'glutes'], ['quads', 'calves']),

    // ---------- 扩充 ----------
    side_lying_leg_raise: m(['glutes'], ['obliques']),
    clamshell: m(['glutes']),
    single_leg_glute_bridge: m(['glutes', 'hamstrings'], ['lower_back']),
    russian_twist: m(['obliques', 'abs'], ['hip_flexors']),
    v_up: m(['abs', 'hip_flexors'], ['obliques']),
    reverse_crunch: m(['abs'], ['hip_flexors']),
    leg_raise: m(['abs', 'hip_flexors']),
    flutter_kicks: m(['abs', 'hip_flexors'], ['quads']),
    hollow_hold: m(['abs'], ['hip_flexors', 'obliques']),
    bear_plank: m(['abs', 'delts'], ['quads', 'triceps']),
    plank_jacks: m(['abs', 'delts'], ['adductors', 'glutes', 'calves']),
    plank_shoulder_tap: m(['abs', 'obliques'], ['delts', 'triceps']),
    donkey_kick: m(['glutes'], ['hamstrings']),
    fire_hydrant: m(['glutes'], ['obliques']),
    reverse_lunge: m(['quads', 'glutes'], ['hamstrings', 'adductors']),
    sumo_squat: m(['quads', 'glutes', 'adductors'], ['hamstrings']),
    wall_pushup: m(['chest', 'triceps'], ['delts']),
    diamond_pushup: m(['triceps', 'chest'], ['delts', 'abs']),
    single_leg_rdl: m(['hamstrings', 'glutes'], ['lower_back', 'calves']),
    side_lunge: m(['adductors', 'quads', 'glutes'], ['hamstrings']),
    curtsy_lunge: m(['glutes', 'quads'], ['adductors']),
    kb_deadlift: m(['hamstrings', 'glutes'], ['lower_back', 'forearms', 'quads']),
    kb_press: m(['delts', 'triceps'], ['traps', 'abs']),
    kb_row: m(['lats', 'biceps'], ['traps', 'delts']),
    kb_goblet_reverse_lunge: m(['quads', 'glutes'], ['hamstrings', 'abs']),
    chin_up: m(['lats', 'biceps'], ['forearms', 'traps']),
    dead_hang: m(['forearms', 'lats'], ['delts']),
    negative_pullup: m(['lats', 'biceps'], ['forearms', 'traps']),
    jump_rope_alt: m(['calves'], ['quads', 'hamstrings', 'forearms']),
    jump_rope_side: m(['calves'], ['quads', 'adductors']),
    jump_rope_high_knee: m(['calves', 'hip_flexors', 'quads'], ['abs']),
    band_curl: m(['biceps'], ['forearms']),
    band_overhead_press: m(['delts', 'triceps'], ['traps']),
    db_reverse_fly: m(['delts', 'traps'], ['lats']),
    db_triceps_extension: m(['triceps']),
    db_front_raise: m(['delts'], ['traps']),
    neck_stretch: m(['traps']),
    shoulder_rolls: m(['traps', 'delts']),
    thread_the_needle: m(['traps', 'delts'], ['lats']),
    butterfly_stretch: m(['adductors'], ['hip_flexors']),
    knee_to_chest: m(['lower_back', 'glutes'], ['hamstrings']),
};
