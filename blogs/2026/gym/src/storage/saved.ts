import type { Conditions, SavedWorkout, Workout } from '../types';

const SAVED_KEY = 'gym.saved.v1';
const LAST_CONDITIONS_KEY = 'gym.lastConditions.v1';

// 只依赖 getItem/setItem，测试时可传入内存实现
export type KeyValueStore = Pick<Storage, 'getItem' | 'setItem'>;

function readJson<T>(store: KeyValueStore, key: string, fallback: T): T {
    try {
        const raw = store.getItem(key);
        if (raw === null)
            return fallback;
        return JSON.parse(raw) as T;
    } catch {
        return fallback;
    }
}

function writeJson(store: KeyValueStore, key: string, value: unknown): void {
    try {
        store.setItem(key, JSON.stringify(value));
    } catch {
        // 隐私模式或存储已满时静默失败，页面仍可使用
    }
}

function newId(): string {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createSavedStore(store: KeyValueStore) {
    function list(): SavedWorkout[] {
        return readJson<SavedWorkout[]>(store, SAVED_KEY, []);
    }

    function write(items: SavedWorkout[]): void {
        writeJson(store, SAVED_KEY, items);
    }

    return {
        list,

        save(name: string, workout: Workout): SavedWorkout {
            const item: SavedWorkout = { id: newId(), name, workout, createdAt: new Date().toISOString() };
            write([item, ...list()]);
            return item;
        },

        remove(id: string): void {
            write(list().filter((s) => s.id !== id));
        },

        rename(id: string, name: string): void {
            write(list().map((s) => (s.id === id ? { ...s, name } : s)));
        },

        markUsed(id: string): void {
            const now = new Date().toISOString();
            write(list().map((s) => (s.id === id ? { ...s, lastUsedAt: now } : s)));
        },

        lastConditions(): Conditions | null {
            return readJson<Conditions | null>(store, LAST_CONDITIONS_KEY, null);
        },

        rememberConditions(c: Conditions): void {
            writeJson(store, LAST_CONDITIONS_KEY, c);
        },
    };
}

export type SavedStore = ReturnType<typeof createSavedStore>;
