import { useEffect, useMemo, useState } from 'react';
import { EXERCISES, EXERCISE_BY_ID } from '../data/exercises';
import { EXERCISE_MUSCLES, MUSCLE_DETAIL_LABEL } from '../data/muscles';
import { CATEGORY_LABEL, EQUIPMENT_LABEL } from '../labels';
import type { Category, Equipment, Exercise, Gender, Step } from '../types';
import { ExerciseBodyMap } from './BodyMap';
import { ExerciseAnimation } from './ExerciseAnimation';

interface Props {
    favorites: string[];
    coach: Gender;
    canAdd: boolean;                // 有正在查看的训练时才能"加入当前训练"
    addedIds: string[];             // 当前训练里已有的动作
    onToggleFavorite: (id: string) => void;
    onAdd: (id: string) => void;
    onBack: () => void;
}

const PHASE_LABEL = { warmup: '热身', main: '主训练', cooldown: '放松' } as const;

function dose(e: Exercise): string {
    const amount = e.mode === 'time' ? `${e.defaultSeconds} 秒` : `${e.defaultReps} 次`;
    return e.perSide ? `${amount}（每侧）` : amount;
}

function tags(e: Exercise): string[] {
    const out = [
        e.equipment.length > 0 ? e.equipment.map((q) => EQUIPMENT_LABEL[q]).join('、') : '徒手',
        e.phases.map((p) => PHASE_LABEL[p]).join(' / '),
        `出汗 ${e.sweat} 级`,
        `强度 ${e.intensity} 级`,
    ];
    if (e.locations.length === 1)
        out.push(e.locations[0] === 'indoor' ? '室内' : '室外');
    if (e.noisy)
        out.push('有噪音');
    if (e.needsShoes)
        out.push('需要运动鞋');
    if (e.needsSpace)
        out.push('需要空间');
    return out;
}

// 动作库里的动画预览：自己计时，循环播放；单侧动作示范左侧
function AnimationPreview({ exerciseId, coach }: { exerciseId: string; coach: Gender }) {
    const [t, setT] = useState(0);
    useEffect(() => {
        const start = performance.now();
        const id = window.setInterval(() => setT((performance.now() - start) / 1000), 100);
        return () => window.clearInterval(id);
    }, [exerciseId]);
    const step: Step = {
        kind: 'work',
        exerciseId,
        seconds: 3600,
        label: '',
        blockIndex: 0,
        side: EXERCISE_BY_ID[exerciseId].perSide ? 'left' : undefined,
    };
    return <ExerciseAnimation step={step} intoStep={t} paused={false} coach={coach} />;
}

type EquipFilter = 'all' | 'none' | Equipment;

export function ExerciseLibrary({ favorites, coach, canAdd, addedIds, onToggleFavorite, onAdd, onBack }: Props) {
    const [query, setQuery] = useState('');
    const [category, setCategory] = useState<Category | 'all'>('all');
    const [equip, setEquip] = useState<EquipFilter>('all');
    const [onlyFav, setOnlyFav] = useState(false);
    const [openId, setOpenId] = useState<string | null>(null);
    const [message, setMessage] = useState('');

    const list = useMemo(() => EXERCISES.filter((e) => {
        if (query && !e.name.includes(query.trim()))
            return false;
        if (category !== 'all' && e.category !== category)
            return false;
        if (equip === 'none' && e.equipment.length > 0)
            return false;
        if (equip !== 'all' && equip !== 'none' && !e.equipment.includes(equip))
            return false;
        if (onlyFav && !favorites.includes(e.id))
            return false;
        return true;
    }), [query, category, equip, onlyFav, favorites]);

    function add(e: Exercise) {
        onAdd(e.id);
        setMessage(`已加入当前训练：${e.name}`);
    }

    return (
        <section className="card">
            <div className="row space">
                <h2>动作库（{EXERCISES.length} 个）</h2>
                <button type="button" onClick={onBack}>返回</button>
            </div>
            <p className="muted small">
                点动作看示范动画和练到的肌肉。♥ 喜欢的动作在生成训练时会优先选用（仍按当天条件过滤）
                {canAdd ? '；＋ 可以把动作加入当前训练。' : '；先生成一次训练，就能把动作加入训练。'}
            </p>

            <div className="library-filters">
                <input
                    type="search"
                    placeholder="搜索动作名"
                    aria-label="搜索动作名"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                />
                <div className="chips">
                    {(['all', ...Object.keys(CATEGORY_LABEL)] as (Category | 'all')[]).map((c) => (
                        <button key={c} type="button" className={category === c ? 'chip on' : 'chip'} aria-pressed={category === c} onClick={() => setCategory(c)}>
                            {c === 'all' ? '全部' : CATEGORY_LABEL[c]}
                        </button>
                    ))}
                    <button type="button" className={onlyFav ? 'chip on' : 'chip'} aria-pressed={onlyFav} onClick={() => setOnlyFav(!onlyFav)}>
                        ♥ 只看喜欢的
                    </button>
                </div>
                <div className="chips">
                    {(['all', 'none', ...Object.keys(EQUIPMENT_LABEL)] as EquipFilter[]).map((q) => (
                        <button key={q} type="button" className={equip === q ? 'chip on' : 'chip'} aria-pressed={equip === q} onClick={() => setEquip(q)}>
                            {q === 'all' ? '全部器械' : q === 'none' ? '徒手' : EQUIPMENT_LABEL[q]}
                        </button>
                    ))}
                </div>
            </div>

            {message && <p className="notice">{message}</p>}
            <p className="muted small">共 {list.length} 个</p>

            <ul className="library">
                {list.map((e) => {
                    const fav = favorites.includes(e.id);
                    const open = openId === e.id;
                    const added = addedIds.includes(e.id);
                    return (
                        <li key={e.id} className={open ? 'open' : undefined}>
                            <div className="library-row">
                                <button type="button" className="library-main" aria-expanded={open} onClick={() => setOpenId(open ? null : e.id)}>
                                    <span className="item-name">{e.name}</span>
                                    <span className="muted small">
                                        {CATEGORY_LABEL[e.category]} · {e.equipment.length > 0 ? e.equipment.map((q) => EQUIPMENT_LABEL[q]).join('、') : '徒手'} · {dose(e)}
                                    </span>
                                    <span className="muted small">
                                        {EXERCISE_MUSCLES[e.id].primary.map((m) => MUSCLE_DETAIL_LABEL[m]).join(' · ')}
                                    </span>
                                </button>
                                <div className="row">
                                    <button
                                        type="button"
                                        className={fav ? 'fav on' : 'fav'}
                                        aria-pressed={fav}
                                        aria-label={`喜欢 ${e.name}`}
                                        onClick={() => onToggleFavorite(e.id)}
                                    >
                                        {fav ? '♥' : '♡'}
                                    </button>
                                    {canAdd && (
                                        <button type="button" className="primary" aria-label={`加入当前训练 ${e.name}`} onClick={() => add(e)}>
                                            {added ? '＋ 再加' : '＋ 加入'}
                                        </button>
                                    )}
                                </div>
                            </div>
                            {open && (
                                <div className="library-detail">
                                    <AnimationPreview exerciseId={e.id} coach={coach} />
                                    <ExerciseBodyMap exerciseId={e.id} stretch={e.category === 'stretch'} />
                                    <ul className="cues-list small">
                                        {e.cues.map((c) => (
                                            <li key={c}>{c}</li>
                                        ))}
                                    </ul>
                                    <div className="tags">
                                        {tags(e).map((t) => (
                                            <span key={t} className="tag">{t}</span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </li>
                    );
                })}
            </ul>
        </section>
    );
}
