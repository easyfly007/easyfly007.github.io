import { useEffect, useMemo, useState } from 'react';
import { ConditionsForm } from './components/ConditionsForm';
import { ExerciseLibrary } from './components/ExerciseLibrary';
import { ProfilePage, ProfilePrompt } from './components/ProfilePage';
import { Player } from './components/Player';
import { Report } from './components/Report';
import { SavedList } from './components/SavedList';
import { SettingsCard } from './components/SettingsCard';
import { WorkoutView } from './components/WorkoutView';
import { addExercise, removeItem } from './engine/edit';
import { generateWorkout } from './engine/generate';
import { newSeed } from './engine/random';
import { MIN_RECORD_SEC, playedSeconds } from './engine/report';
import { describeConditions } from './labels';
import { setVoiceGender, unlockAudio } from './speech';
import { buildProfile, profileText } from './storage/profile';
import { writeBoundFile } from './storage/profileFile';
import { createSavedStore } from './storage/saved';
import type { Conditions, SavedWorkout, Settings, Workout, WorkoutRecord } from './types';

const DEFAULT_CONDITIONS: Conditions = {
    equipment: ['mat'],
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

type View = 'home' | 'plan' | 'player' | 'report' | 'library' | 'profile';

function App() {
    const store = useMemo(() => createSavedStore(window.localStorage), []);
    const [conditions, setConditions] = useState<Conditions>(() => store.lastConditions() ?? DEFAULT_CONDITIONS);
    const [saved, setSaved] = useState<SavedWorkout[]>(() => store.list());
    const [workout, setWorkout] = useState<Workout | null>(null);
    const [savedId, setSavedId] = useState<string | null>(null);
    const [view, setView] = useState<View>('home');
    const [history, setHistory] = useState<WorkoutRecord[]>(() => store.history());
    const [report, setReport] = useState<WorkoutRecord | null>(null);
    const [weightKg, setWeightKg] = useState<number | undefined>(() => store.weightKg());
    const [startedAt, setStartedAt] = useState('');
    const [settings, setSettings] = useState<Settings>(() => store.settings());
    const [notice, setNotice] = useState<string | null>(null);
    const [favorites, setFavorites] = useState<string[]>(() => store.favorites());
    const [libraryFrom, setLibraryFrom] = useState<'home' | 'plan'>('home');
    // 训练后同步档案文件的结果；askProfile：第一次训练后提示保存档案
    const [sync, setSync] = useState<'ok' | 'need-permission' | null>(null);
    const [askProfile, setAskProfile] = useState(false);

    useEffect(() => {
        setVoiceGender(settings.voice);
    }, [settings.voice]);

    function changeSettings(s: Settings) {
        store.saveSettings(s);
        setSettings(s);
    }

    const current = saved.find((s) => s.id === savedId);

    function generate(c: Conditions) {
        store.rememberConditions(c);
        setConditions(c);
        setWorkout(generateWorkout(c, newSeed(), favorites));
        setSavedId(null);
        setNotice(null);
        setView('plan');
    }

    function open(s: SavedWorkout) {
        setWorkout(s.workout);
        setConditions(s.workout.conditions);
        setSavedId(s.id);
        setNotice(null);
        setView('plan');
    }

    function start(s?: SavedWorkout) {
        unlockAudio();
        if (s) {
            setWorkout(s.workout);
            setSavedId(s.id);
        }
        const id = s?.id ?? savedId;
        if (id) {
            store.markUsed(id);
            setSaved(store.list());
        }
        setStartedAt(new Date().toISOString());
        setNotice(null);
        setView('player');
    }

    // 跟练结束：练够 1 分钟才记录并显示报告，否则回到计划页
    function finish(played: [number, number][]) {
        if (!workout)
            return;
        const draft = { workout, name: current?.name, startedAt, endedAt: new Date().toISOString(), played };
        const sec = Math.round(playedSeconds({ id: '', ...draft }));
        if (sec < MIN_RECORD_SEC) {
            setNotice(`这次实际练了 ${sec} 秒（跳过的部分不算），不足 ${MIN_RECORD_SEC} 秒，没有记录，也不生成训练报告。`);
            setView('plan');
            return;
        }
        const r = store.addRecord(draft);
        setHistory(store.history());
        setReport(r);
        setView('report');
        syncProfile(false);
    }

    // 训练记录同步到档案文件：没绑定文件且从没填过名字时，提示保存档案
    function syncProfile(interactive: boolean) {
        const info = store.profileInfo();
        setSync(null);
        setAskProfile(info.name === undefined);
        if (!info.fileName)
            return;
        writeBoundFile(profileText(buildProfile(store, info.name ?? '')), interactive)
            .then((res) => {
                if (res.status === 'ok') {
                    store.saveProfileInfo({ ...store.profileInfo(), lastSyncAt: new Date().toISOString() });
                    setSync('ok');
                } else if (res.status === 'need-permission')
                    setSync('need-permission');
            })
            .catch(() => setSync('need-permission'));
    }

    // 档案合并后重新读取本机数据
    function reloadLocal() {
        setHistory(store.history());
        setFavorites(store.favorites());
        setWeightKg(store.weightKg());
        setSettings(store.settings());
    }

    function openRecord(r: WorkoutRecord) {
        setReport(r);
        setView('report');
    }

    function toggleFavorite(id: string) {
        const next = favorites.includes(id) ? favorites.filter((x) => x !== id) : [...favorites, id];
        store.saveFavorites(next);
        setFavorites(next);
    }

    function openLibrary(from: 'home' | 'plan') {
        setLibraryFrom(from);
        setView('library');
    }

    // 手动改过的训练不再等同于原模板：清掉模板关联，用户可另存为新模板
    function editWorkout(w: Workout) {
        setWorkout(w);
        setSavedId(null);
        setNotice(null);
    }

    function changeWeight(kg: number) {
        store.setWeightKg(kg);
        setWeightKg(kg);
    }

    function save() {
        if (!workout)
            return;
        const name = window.prompt('给这个训练起个名字', describeConditions(workout.conditions));
        if (!name)
            return;
        const s = store.save(name.trim(), workout);
        setSaved(store.list());
        setSavedId(s.id);
    }

    function remove(s: SavedWorkout) {
        if (!window.confirm(`删除模板"${s.name}"？`))
            return;
        store.remove(s.id);
        setSaved(store.list());
    }

    if (view === 'player' && workout)
        return <Player workout={workout} coach={settings.coach} onExit={finish} />;

    return (
        <main>
            <header className="app-header row space">
                <div>
                    <h1>健身计划生成器</h1>
                    <p className="muted">告诉我你今天的条件，生成一节课，跟着做就好</p>
                </div>
                <button type="button" onClick={() => setView('profile')}>👤 我的</button>
            </header>

            {view === 'profile' ? (
                <ProfilePage
                    store={store}
                    records={history}
                    weightKg={weightKg}
                    onChanged={reloadLocal}
                    onOpenRecord={openRecord}
                    onBack={() => setView('home')}
                />
            ) : view === 'library' ? (
                <ExerciseLibrary
                    favorites={favorites}
                    coach={settings.coach}
                    canAdd={workout !== null}
                    addedIds={workout ? workout.blocks.flatMap((b) => b.items.map((i) => i.exerciseId)) : []}
                    onToggleFavorite={toggleFavorite}
                    onAdd={(id) => workout && editWorkout(addExercise(workout, id))}
                    onBack={() => setView(libraryFrom === 'plan' && workout ? 'plan' : 'home')}
                />
            ) : view === 'report' && report ? (
                <>
                    {sync === 'ok' && <p className="notice">已同步到档案文件</p>}
                    {sync === 'need-permission' && (
                        <p className="notice">
                            训练记录已保存在本机，但浏览器需要你允许才能写入档案文件。
                            <button type="button" onClick={() => syncProfile(true)}>同步到档案文件</button>
                        </p>
                    )}
                    {askProfile && <ProfilePrompt store={store} onDone={() => {
                        setAskProfile(false);
                        setSync(store.profileInfo().fileName ? 'ok' : null);
                    }}
                    />}
                    <Report
                    record={report}
                    history={history}
                    weightKg={weightKg}
                    onWeightChange={changeWeight}
                    onBack={() => setView('home')}
                    />
                </>
            ) : view === 'plan' && workout ? (
                <>
                    {notice && <p className="notice">{notice}</p>}
                    <WorkoutView
                        workout={workout}
                        title={describeConditions(workout.conditions)}
                        savedName={current?.name}
                        onStart={() => start()}
                        onRegenerate={() => generate(workout.conditions)}
                        onSave={save}
                        onBack={() => setView('home')}
                        onOpenLibrary={() => openLibrary('plan')}
                        onRemoveItem={(bi, i) => editWorkout(removeItem(workout, bi, i))}
                    />
                </>
            ) : (
                <>
                    <ConditionsForm value={conditions} onChange={setConditions} onGenerate={() => generate(conditions)} />
                    <section className="card row space">
                        <div>
                            <h2>动作库</h2>
                            <p className="muted small">浏览全部动作的示范动画和练到的肌肉，标记喜欢的动作{favorites.length > 0 ? `（已喜欢 ${favorites.length} 个）` : ''}</p>
                        </div>
                        <button type="button" onClick={() => openLibrary('home')}>浏览动作库</button>
                    </section>
                    <SavedList
                        items={saved}
                        onOpen={open}
                        onStart={start}
                        onRegenerate={(s) => generate(s.workout.conditions)}
                        onDelete={remove}
                    />
                    <SettingsCard value={settings} onChange={changeSettings} />
                </>
            )}
        </main>
    );
}

export default App;
