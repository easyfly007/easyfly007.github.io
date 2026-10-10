import { useMemo, useState } from 'react';
import { ConditionsForm } from './components/ConditionsForm';
import { Player } from './components/Player';
import { SavedList } from './components/SavedList';
import { WorkoutView } from './components/WorkoutView';
import { generateWorkout } from './engine/generate';
import { newSeed } from './engine/random';
import { describeConditions } from './labels';
import { unlockAudio } from './speech';
import { createSavedStore } from './storage/saved';
import type { Conditions, SavedWorkout, Workout } from './types';

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
};

type View = 'home' | 'plan' | 'player';

function App() {
    const store = useMemo(() => createSavedStore(window.localStorage), []);
    const [conditions, setConditions] = useState<Conditions>(() => store.lastConditions() ?? DEFAULT_CONDITIONS);
    const [saved, setSaved] = useState<SavedWorkout[]>(() => store.list());
    const [workout, setWorkout] = useState<Workout | null>(null);
    const [savedId, setSavedId] = useState<string | null>(null);
    const [view, setView] = useState<View>('home');

    const current = saved.find((s) => s.id === savedId);

    function generate(c: Conditions) {
        store.rememberConditions(c);
        setConditions(c);
        setWorkout(generateWorkout(c, newSeed()));
        setSavedId(null);
        setView('plan');
    }

    function open(s: SavedWorkout) {
        setWorkout(s.workout);
        setConditions(s.workout.conditions);
        setSavedId(s.id);
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
        setView('player');
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
        return <Player workout={workout} onExit={() => setView('plan')} />;

    return (
        <main>
            <header className="app-header">
                <h1>健身计划生成器</h1>
                <p className="muted">告诉我你今天的条件，生成一节课，跟着做就好</p>
            </header>

            {view === 'plan' && workout ? (
                <WorkoutView
                    workout={workout}
                    title={describeConditions(workout.conditions)}
                    savedName={current?.name}
                    onStart={() => start()}
                    onRegenerate={() => generate(workout.conditions)}
                    onSave={save}
                    onBack={() => setView('home')}
                />
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
                </>
            )}
        </main>
    );
}

export default App;
