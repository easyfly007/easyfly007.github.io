import type { Conditions, Exercise, Phase } from '../types';

// 不能洗澡时的出汗上限；能洗澡时按强度放开
export function maxSweat(c: Conditions): number {
    if (!c.canShower)
        return 2;
    return [0, 3, 4, 5][c.intensity];
}

export function maxIntensity(c: Conditions): number {
    return [0, 2, 4, 5][c.intensity];
}

// 动作在当前条件下是否可做（与阶段无关的硬性约束）
export function isFeasible(e: Exercise, c: Conditions): boolean {
    if (!e.equipment.every((eq) => c.equipment.includes(eq)))
        return false;
    if (!e.locations.includes(c.location))
        return false;
    if (e.needsSpace && c.space === 'small')
        return false;
    if (e.noisy && c.quiet)
        return false;
    if (e.needsShoes && !c.hasShoes)
        return false;
    if (e.stresses.some((s) => c.limitations.includes(s)))
        return false;
    return true;
}

// 某阶段的候选动作。热身和放松只看可行性和出汗；主训练还受强度上限约束
export function candidates(all: Exercise[], c: Conditions, phase: Phase): Exercise[] {
    const sweatCap = phase === 'main' ? maxSweat(c) : Math.min(maxSweat(c), 2);
    return all.filter((e) => {
        if (!e.phases.includes(phase))
            return false;
        if (!isFeasible(e, c))
            return false;
        if (e.sweat > sweatCap)
            return false;
        if (phase === 'main' && e.intensity > maxIntensity(c))
            return false;
        return true;
    });
}
