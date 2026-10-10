// 浏览器里读写档案文件。
// 电脑上的 Chrome / Edge 支持"文件系统访问"：用户第一次选好位置后，记住这个文件（存在 IndexedDB），
// 之后每次都能直接读写同一个文件（浏览器可能会请用户确认一次权限）。
// 手机和其他浏览器不支持：只能"导出"下载一个文件、"导入"选一个文件

type PermissionState = 'granted' | 'denied' | 'prompt';
interface Handle extends FileSystemFileHandle {
    queryPermission?: (o: { mode: 'readwrite' }) => Promise<PermissionState>;
    requestPermission?: (o: { mode: 'readwrite' }) => Promise<PermissionState>;
}
interface PickerWindow {
    showSaveFilePicker?: (o: object) => Promise<Handle>;
    showOpenFilePicker?: (o: object) => Promise<Handle[]>;
}

const DB_NAME = 'gym-files';
const STORE = 'handles';
const KEY = 'profile';
const TYPES = [{ description: '训练档案', accept: { 'application/json': ['.json'] } }];

function picker(): PickerWindow {
    return window as unknown as PickerWindow;
}

export function fileAccessSupported(): boolean {
    return typeof window !== 'undefined' && typeof picker().showSaveFilePicker === 'function' && typeof indexedDB !== 'undefined';
}

function db(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(STORE);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function getHandle(): Promise<Handle | undefined> {
    const d = await db();
    return new Promise((resolve, reject) => {
        const req = d.transaction(STORE).objectStore(STORE).get(KEY);
        req.onsuccess = () => resolve(req.result as Handle | undefined);
        req.onerror = () => reject(req.error);
    });
}

async function setHandle(h: Handle): Promise<void> {
    const d = await db();
    return new Promise((resolve, reject) => {
        const tx = d.transaction(STORE, 'readwrite');
        tx.objectStore(STORE).put(h, KEY);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

// 确认对已绑定文件的读写权限；interactive 时可以弹出浏览器的确认（需要在用户点击后调用）
async function permitted(h: Handle, interactive: boolean): Promise<boolean> {
    const opts = { mode: 'readwrite' } as const;
    if ((await h.queryPermission?.(opts)) === 'granted')
        return true;
    if (!interactive)
        return false;
    return (await h.requestPermission?.(opts)) === 'granted';
}

async function write(h: Handle, text: string): Promise<void> {
    const w = await h.createWritable();
    await w.write(text);
    await w.close();
}

// 新建档案文件（让用户选位置，默认在"文档"里），返回文件名
export async function createProfileFile(suggestedName: string, text: string): Promise<string> {
    const h = await picker().showSaveFilePicker!({ suggestedName, startIn: 'documents', types: TYPES });
    await write(h, text);
    await setHandle(h);
    return h.name;
}

// 打开已有的档案文件并记住它，返回文件名和内容
export async function openProfileFile(): Promise<{ name: string; text: string }> {
    const [h] = await picker().showOpenFilePicker!({ startIn: 'documents', types: TYPES, multiple: false });
    await setHandle(h);
    return { name: h.name, text: await (await h.getFile()).text() };
}

export type FileResult = { status: 'ok'; name: string; text?: string } | { status: 'none' } | { status: 'need-permission' };

export async function readBoundFile(interactive: boolean): Promise<FileResult> {
    if (!fileAccessSupported())
        return { status: 'none' };
    const h = await getHandle();
    if (!h)
        return { status: 'none' };
    if (!(await permitted(h, interactive)))
        return { status: 'need-permission' };
    return { status: 'ok', name: h.name, text: await (await h.getFile()).text() };
}

export async function writeBoundFile(text: string, interactive: boolean): Promise<FileResult> {
    if (!fileAccessSupported())
        return { status: 'none' };
    const h = await getHandle();
    if (!h)
        return { status: 'none' };
    if (!(await permitted(h, interactive)))
        return { status: 'need-permission' };
    await write(h, text);
    return { status: 'ok', name: h.name };
}

// 不支持文件系统访问时：保存一份文件 / 选择文件。
// 手机上优先用系统分享面板（iPhone 可"存储到文件"，选文件夹，同名时系统会问是否替换）；否则直接下载
export async function downloadFile(name: string, text: string): Promise<'shared' | 'downloaded'> {
    const file = new File([text], name, { type: 'application/json' });
    const mobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    if (mobile && navigator.canShare?.({ files: [file] })) {
        await navigator.share({ files: [file], title: name });
        return 'shared';
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    return 'downloaded';
}

export function chooseFile(): Promise<string | null> {
    return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json,application/json';
        input.onchange = async () => {
            const f = input.files?.[0];
            resolve(f ? await f.text() : null);
        };
        input.click();
    });
}

export function profileFileName(name: string): string {
    return `健身档案-${name || '我'}.json`;
}
