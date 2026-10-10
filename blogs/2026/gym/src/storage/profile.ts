import type { Settings, WorkoutRecord } from '../types';
import type { SavedStore } from './saved';

// 训练档案文件：统一格式的 JSON，包含名字、训练记录、喜欢的动作、设置和体重。
// 文件只在用户本地（自己选的位置或下载的文件），不经过任何服务器

export const PROFILE_FORMAT = 'gym-profile';

export interface ProfileFile {
    format: typeof PROFILE_FORMAT;
    version: 1;
    name: string;
    updatedAt: string;
    weightKg?: number;
    settings?: Settings;
    favorites: string[];
    records: WorkoutRecord[];
}

export function buildProfile(store: SavedStore, name: string): ProfileFile {
    return {
        format: PROFILE_FORMAT,
        version: 1,
        name,
        updatedAt: new Date().toISOString(),
        weightKg: store.weightKg(),
        settings: store.settings(),
        favorites: store.favorites(),
        records: store.history(),
    };
}

export function profileText(p: ProfileFile): string {
    return JSON.stringify(p, null, 2);
}

export function parseProfile(text: string): ProfileFile {
    let data: unknown;
    try {
        data = JSON.parse(text);
    } catch {
        throw new Error('文件不是有效的 JSON');
    }
    const p = data as Partial<ProfileFile>;
    if (p?.format !== PROFILE_FORMAT || !Array.isArray(p.records))
        throw new Error('这不是健身计划生成器的训练档案文件');
    return {
        format: PROFILE_FORMAT,
        version: 1,
        name: typeof p.name === 'string' ? p.name : '',
        updatedAt: typeof p.updatedAt === 'string' ? p.updatedAt : '',
        weightKg: typeof p.weightKg === 'number' ? p.weightKg : undefined,
        settings: p.settings,
        favorites: Array.isArray(p.favorites) ? p.favorites.filter((x): x is string => typeof x === 'string') : [],
        records: p.records.filter((r) => r && typeof r.id === 'string' && typeof r.startedAt === 'string' && Array.isArray(r.played)),
    };
}

// 把档案文件合并进本机：训练记录按 id 取并集（新的在前），喜欢的动作取并集，本机没有的体重/设置用文件里的
export function mergeProfile(store: SavedStore, p: ProfileFile): { added: number } {
    const local = store.history();
    const ids = new Set(local.map((r) => r.id));
    const incoming = p.records.filter((r) => !ids.has(r.id));
    const all = [...local, ...incoming].sort((a, b) => (a.startedAt < b.startedAt ? 1 : -1));
    store.replaceHistory(all);
    store.saveFavorites([...new Set([...store.favorites(), ...p.favorites])]);
    if (store.weightKg() === undefined && p.weightKg !== undefined)
        store.setWeightKg(p.weightKg);
    if (!store.hasSettings() && p.settings)
        store.saveSettings(p.settings);
    if (p.name)
        store.saveProfileInfo({ ...store.profileInfo(), name: p.name });
    return { added: incoming.length };
}
