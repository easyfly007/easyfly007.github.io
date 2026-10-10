import { useEffect, useMemo, useState } from 'react';
import { ConditionsForm } from './components/ConditionsForm';
import { History } from './components/History';
import { Player } from './components/Player';
import { Report } from './components/Report';
import { SavedList } from './components/SavedList';
import { SettingsCard } from './components/SettingsCard';
import { WorkoutView } from './components/WorkoutView';
import { generateWorkout } from './engine/generate';
import { newSeed } from './engine/random';
import { MIN_RECORD_SEC, playedSeconds } from './engine/report';
import { describeConditions } from './labels';
import { setVoiceGender, unlockAudio } from './speech';
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

type View = 'home' | 'plan' | 'player' | 'report';

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
        setWorkout(generateWorkout(c, newSeed()));
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
    }

    function openRecord(r: WorkoutRecord) {
        setReport(r);
        setView('report');
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
            <header className="app-header">
                <h1>健身计划生成器</h1>
                <p className="muted">告诉我你今天的条件，生成一节课，跟着做就好</p>
            </header>

            {view === 'report' && report ? (
                <Report
                    record={report}
                    history={history}
                    weightKg={weightKg}
                    onWeightChange={changeWeight}
                    onBack={() => setView('home')}
                />
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
                    />
                </>
            ) : (
                <>
                    <ConditionsForm value={conditions} onChange={setConditions} onGenerate={() => generate(conditions)} />
                    <SavedList
                        items={saved}
                        onOpen={open}
                        onStart={start}
                        onRegenerate={(s) => generate(s.workout.conditions)}
                        onDelete={remove}
                    />
                    <SettingsCard value={settings} onChange={changeSettings} />
                    <History records={history} onOpen={openRecord} />
                </>
            )}
        </main>
    );
}

export default App;
