import type { Conditions } from '../types';

// 测试用的基准条件：室内、徒手、30 分钟、中等强度
export const BASE: Conditions = {
    equipment: [],
    location: 'indoor',
    space: 'small',
    quiet: false,
    hasShoes: true,
    durationMin: 30,
    intensity: 2,
    canShower: true,
    goal: 'general',
    limitations: [],
    rest: 'normal',
};
