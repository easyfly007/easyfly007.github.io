import { useEffect, useState } from 'react';
import { shortDateTime } from '../labels';
import { buildProfile, mergeProfile, parseProfile, profileText } from '../storage/profile';
import {
    chooseFile,
    createProfileFile,
    downloadFile,
    fileAccessSupported,
    openProfileFile,
    profileFileName,
    readBoundFile,
    writeBoundFile,
} from '../storage/profileFile';
import type { SavedStore } from '../storage/saved';
import type { WorkoutRecord } from '../types';
import { History } from './History';
import { MonthCalendar } from './MonthCalendar';

interface Props {
    store: SavedStore;
    records: WorkoutRecord[];
    weightKg?: number;
    onChanged: () => void;          // 档案合并后让上层重新读取本机数据
    onOpenRecord: (r: WorkoutRecord) => void;
    onBack: () => void;
}

function errorText(e: unknown): string {
    if (e instanceof DOMException && e.name === 'AbortError')
        return '已取消';
    return e instanceof Error ? e.message : String(e);
}

// "我的"：名字、档案文件（读写/导入导出）、训练日历与本月总结、历史趋势
export function ProfilePage({ store, records, weightKg, onChanged, onOpenRecord, onBack }: Props) {
    const [info, setInfo] = useState(() => store.profileInfo());
    const [nameInput, setNameInput] = useState(info.name ?? '');
    const [editing, setEditing] = useState(!info.name);
    const [status, setStatus] = useState('');
    const [needPermission, setNeedPermission] = useState(false);
    const supported = fileAccessSupported();

    function updateInfo(patch: Partial<typeof info>) {
        const next = { ...store.profileInfo(), ...patch };
        store.saveProfileInfo(next);
        setInfo(next);
    }

    function absorb(text: string, from: string) {
        const p = parseProfile(text);
        const { added } = mergeProfile(store, p);
        setInfo(store.profileInfo());
        setNameInput(store.profileInfo().name ?? '');
        onChanged();
        setStatus(`已载入${from}，新增 ${added} 条训练记录`);
    }

    // 打开"我的"时：如果绑定过档案文件，读入并合并
    useEffect(() => {
        if (!supported)
            return;
        let alive = true;
        readBoundFile(true)
            .then((r) => {
                if (!alive)
                    return;
                if (r.status === 'ok' && r.text !== undefined) {
                    absorb(r.text, `档案文件 ${r.name}`);
                    updateInfo({ fileName: r.name });
                } else if (r.status === 'need-permission')
                    setNeedPermission(true);
            })
            .catch((e) => alive && setStatus(`读取档案文件失败：${errorText(e)}`));
        return () => {
            alive = false;
        };
        // 只在进入页面时读一次
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    function saveName() {
        const name = nameInput.trim();
        if (!name)
            return;
        updateInfo({ name });
        setEditing(false);
    }

    const text = () => profileText(buildProfile(store, store.profileInfo().name ?? ''));

    async function run(task: () => Promise<void>) {
        try {
            await task();
        } catch (e) {
            setStatus(errorText(e));
        }
    }

    const create = () => run(async () => {
        const name = await createProfileFile(profileFileName(info.name ?? ''), text());
        updateInfo({ fileName: name, lastSyncAt: new Date().toISOString() });
        setStatus(`已创建档案文件 ${name}，以后每次训练完会自动同步`);
    });

    const openExisting = () => run(async () => {
        const f = await openProfileFile();
        absorb(f.text, `档案文件 ${f.name}`);
        updateInfo({ fileName: f.name });
        // 合并后写回，文件里也包含本机的记录
        await writeBoundFile(text(), true);
        updateInfo({ lastSyncAt: new Date().toISOString() });
    });

    const syncNow = () => run(async () => {
        const r = await readBoundFile(true);
        if (r.status === 'need-permission') {
            setNeedPermission(true);
            setStatus('需要允许读写档案文件');
            return;
        }
        if (r.status === 'ok' && r.text !== undefined)
            absorb(r.text, `档案文件 ${r.name}`);
        const w = await writeBoundFile(text(), true);
        if (w.status === 'ok') {
            setNeedPermission(false);
            updateInfo({ lastSyncAt: new Date().toISOString() });
            setStatus(`已同步到 ${w.name}`);
        }
    });

    const exportFile = () => run(async () => {
        const how = await downloadFile(profileFileName(info.name ?? ''), text());
        setStatus(how === 'shared' ? '已打开分享面板，可选"存储到文件"' : '已导出档案文件（在浏览器的下载里）');
    });

    const importFile = () => run(async () => {
        const t = await chooseFile();
        if (t !== null)
            absorb(t, '导入的档案文件');
    });

    return (
        <>
            <section className="card">
                <div className="row space">
                    <h2>我的</h2>
                    <button type="button" onClick={onBack}>返回</button>
                </div>

                {editing ? (
                    <div className="row">
                        <input
                            className="text-input"
                            placeholder="你的名字或昵称"
                            aria-label="你的名字"
                            value={nameInput}
                            onChange={(e) => setNameInput(e.target.value)}
                        />
                        <button type="button" className="primary" onClick={saveName}>保存</button>
                    </div>
                ) : (
                    <div className="row space">
                        <div className="profile-name">{info.name}</div>
                        <button type="button" onClick={() => setEditing(true)}>改名</button>
                    </div>
                )}

                <h3 className="section-title">训练档案文件</h3>
                {supported ? (
                    info.fileName ? (
                        <>
                            <p className="small">
                                档案文件：{info.fileName}
                                {info.lastSyncAt && <span className="muted"> · 最近同步 {shortDateTime(info.lastSyncAt)}</span>}
                            </p>
                            <div className="row">
                                <button type="button" className={needPermission ? 'primary' : undefined} onClick={syncNow}>
                                    {needPermission ? '允许读写并同步' : '立即同步'}
                                </button>
                                <button type="button" onClick={openExisting}>换一个档案文件</button>
                            </div>
                        </>
                    ) : (
                        <>
                            <p className="muted small">
                                选一个位置保存档案文件（默认在"文档"里）。以后打开"我的"会自动读取它，每次训练完自动同步。
                            </p>
                            <div className="row">
                                <button type="button" className="primary" disabled={!info.name} onClick={create}>创建档案文件</button>
                                <button type="button" onClick={openExisting}>打开已有档案文件</button>
                            </div>
                            {!info.name && <p className="muted small">先填写名字。</p>}
                        </>
                    )
                ) : (
                    <p className="muted small">
                        这个浏览器（如手机浏览器、Safari）不能直接读写本地文件。训练数据保存在本浏览器里，不会上传；
                        可以"导出档案文件"备份到手机的文件里，换手机或浏览器时"导入"恢复。
                    </p>
                )}
                <div className="row">
                    <button type="button" onClick={exportFile}>导出档案文件</button>
                    <button type="button" onClick={importFile}>导入档案文件</button>
                </div>
                {status && <p className="notice">{status}</p>}
            </section>

            <MonthCalendar records={records} weightKg={weightKg} today={new Date()} onOpen={onOpenRecord} />
            <History records={records} onOpen={onOpenRecord} />
        </>
    );
}

interface PromptProps {
    store: SavedStore;
    onDone: () => void;
}

// 第一次训练结束后：提示填写名字并保存训练档案
export function ProfilePrompt({ store, onDone }: PromptProps) {
    const [name, setName] = useState('');
    const [status, setStatus] = useState('');
    const supported = fileAccessSupported();

    async function save() {
        const n = name.trim();
        if (!n)
            return;
        store.saveProfileInfo({ ...store.profileInfo(), name: n });
        const text = profileText(buildProfile(store, n));
        try {
            if (supported) {
                const file = await createProfileFile(profileFileName(n), text);
                store.saveProfileInfo({ ...store.profileInfo(), fileName: file, lastSyncAt: new Date().toISOString() });
            } else
                await downloadFile(profileFileName(n), text);
            onDone();
        } catch (e) {
            setStatus(errorText(e));
        }
    }

    return (
        <section className="card profile-prompt">
            <h3>保存训练档案</h3>
            <p className="muted small">
                {supported
                    ? '填个名字，选一个位置保存档案文件。以后每次训练完会自动同步到这个文件，打开"我的"就能看到全部历史。'
                    : '填个名字，下载一份档案文件保存好。训练数据一直保存在本浏览器里；换手机或浏览器时可在"我的"里导入这个文件。'}
            </p>
            <div className="row">
                <input className="text-input" placeholder="你的名字或昵称" aria-label="档案名字" value={name} onChange={(e) => setName(e.target.value)} />
                <button type="button" className="primary" onClick={save}>保存档案</button>
                <button type="button" onClick={() => {
                    store.saveProfileInfo({ ...store.profileInfo(), name: store.profileInfo().name ?? '' });
                    onDone();
                }}
                >
                    以后再说
                </button>
            </div>
            {status && <p className="notice">{status}</p>}
        </section>
    );
}
