export type Equipment =
    | 'dumbbell'
    | 'barbell'
    | 'kettlebell'
    | 'band'
    | 'mat'
    | 'pullup_bar'
    | 'jump_rope'
    | 'chair';

export type Location = 'indoor' | 'outdoor';
export type Space = 'small' | 'large';
export type Intensity = 1 | 2 | 3;
export type RestLevel = 'short' | 'normal' | 'long';
export type Goal = 'general' | 'fat_loss' | 'strength' | 'flexibility';
export type Limitation = 'knee' | 'back' | 'shoulder' | 'wrist';
export type Phase = 'warmup' | 'main' | 'cooldown';
export type Category = 'strength' | 'cardio' | 'core' | 'mobility' | 'stretch';
export type Muscle =
    | 'legs'
    | 'glutes'
    | 'chest'
    | 'back'
    | 'shoulders'
    | 'arms'
    | 'core'
    | 'full_body';

export interface Conditions {
    equipment: Equipment[];
    location: Location;
    space: Space;
    quiet: boolean;
    hasShoes: boolean;
    durationMin: number;
    intensity: Intensity;
    canShower: boolean;
    goal: Goal;
    limitations: Limitation[];
    rest?: RestLevel;           // 休息时长档位；缺省为默认档（兼容旧数据）
}

export interface Exercise {
    id: string;
    name: string;
    equipment: Equipment[];
    locations: Location[];
    needsShoes: boolean;
    needsSpace: boolean;
    noisy: boolean;
    phases: Phase[];
    category: Category;
    muscles: Muscle[];
    sweat: number;
    intensity: number;
    stresses: Limitation[];
    mode: 'reps' | 'time';
    defaultReps?: number;
    defaultSeconds?: number;
    secPerRep: number;          // 计次动作每次耗时（秒），用于换算跟练时长
    perSide?: boolean;
    cues: string[];
}

export type BlockFormat = 'straight' | 'circuit' | 'flow';

export interface WorkoutItem {
    exerciseId: string;
    sets: number;
    reps?: number;
    seconds?: number;
    restSec: number;
}

export interface Block {
    phase: Phase;
    title: string;
    format: BlockFormat;
    rounds: number;
    restBetweenRoundsSec: number;
    items: WorkoutItem[];
}

export interface Workout {
    seed: number;
    conditions: Conditions;
    blocks: Block[];
    estimatedMin: number;
}

export interface SavedWorkout {
    id: string;
    name: string;
    workout: Workout;
    createdAt: string;
    lastUsedAt?: string;
}

export interface Step {
    kind: 'prepare' | 'work' | 'rest';
    exerciseId: string;         // work 为当前动作；prepare/rest 为下一个动作（用于预告）
    seconds: number;
    label: string;              // 语音与屏幕上的主文字
    blockIndex: number;
    side?: 'left' | 'right';
    reps?: number;
    set?: { index: number; count: number };
    round?: { index: number; count: number };
}
