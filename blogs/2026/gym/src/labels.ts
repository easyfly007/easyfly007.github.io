import type { Conditions, Equipment, Goal, Limitation, Muscle, RestLevel } from './types';

export const EQUIPMENT_LABEL: Record<Equipment, string> = {
    mat: '瑜伽垫',
    dumbbell: '哑铃',
    barbell: '杠铃',
    kettlebell: '壶铃',
    band: '弹力带',
    pullup_bar: '单杠',
    jump_rope: '跳绳',
    chair: '椅子/凳子',
};

export const GOAL_LABEL: Record<Goal, string> = {
    general: '综合',
    fat_loss: '减脂',
    strength: '力量',
    flexibility: '柔韧放松',
};

export const LIMITATION_LABEL: Record<Limitation, string> = {
    knee: '膝盖',
    back: '腰背',
    shoulder: '肩',
    wrist: '手腕',
};

export const MUSCLE_LABEL: Record<Muscle, string> = {
    legs: '腿',
    glutes: '臀',
    chest: '胸',
    back: '背',
    shoulders: '肩',
    arms: '手臂',
    core: '核心',
    full_body: '全身',
};

export const INTENSITY_LABEL = ['', '低', '中', '高'];

export const REST_LABEL: Record<RestLevel, string> = {
    short: '短一些',
    normal: '默认',
    long: '长一些',
};

export const DURATIONS = [15, 20, 30, 45, 60];

// 用于默认的模板名，如"室内-哑铃+瑜伽垫-30分钟-中强度-不出汗"
export function describeConditions(c: Conditions): string {
    const parts = [c.location === 'indoor' ? '室内' : '室外'];
    parts.push(c.equipment.length > 0 ? c.equipment.map((e) => EQUIPMENT_LABEL[e]).join('+') : '徒手');
    parts.push(`${c.durationMin}分钟`);
    parts.push(`${INTENSITY_LABEL[c.intensity]}强度`);
    if (c.goal !== 'general')
        parts.push(GOAL_LABEL[c.goal]);
    if (!c.canShower)
        parts.push('不出汗');
    if (c.rest === 'short')
        parts.push('休息短');
    if (c.rest === 'long')
        parts.push('休息长');
    return parts.join('-');
}

export function formatDuration(totalSec: number): string {
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
}
