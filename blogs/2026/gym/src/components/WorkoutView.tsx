import { EXERCISE_BY_ID } from '../data/exercises';
import { EXERCISE_MUSCLES, MUSCLE_DETAIL_LABEL } from '../data/muscles';
import type { Block, Workout, WorkoutItem } from '../types';
import { ExerciseBodyMap } from './BodyMap';

interface Props {
    workout: Workout;
    title: string;
    savedName?: string;
    onStart: () => void;
    onRegenerate: () => void;
    onSave: () => void;
    onBack: () => void;
    onOpenLibrary: () => void;
    onRemoveItem: (blockIndex: number, itemIndex: number) => void;
}

function dose(item: WorkoutItem, perSide: boolean): string {
    const amount = item.seconds !== undefined ? `${item.seconds} 秒` : `${item.reps} 次`;
    const side = perSide ? '（每侧）' : '';
    const sets = item.sets > 1 ? `${item.sets} 组 × ` : '';
    return `${sets}${amount}${side}`;
}

function blockSubtitle(b: Block): string {
    if (b.format === 'circuit') {
        const first = b.items[0];
        return `${b.rounds} 轮 · 每个动作 ${first?.seconds ?? 0} 秒，休息 ${first?.restSec ?? 0} 秒 · 轮间休息 ${b.restBetweenRoundsSec} 秒`;
    }
    if (b.format === 'straight')
        return `组间休息 ${b.items[0]?.restSec ?? 0} 秒`;
    return '连续进行';
}

export function WorkoutView({ workout, title, savedName, onStart, onRegenerate, onSave, onBack, onOpenLibrary, onRemoveItem }: Props) {
    return (
        <section className="card">
            <div className="plan-head">
                <div>
                    <h2>{savedName ?? '为你生成的训练'}</h2>
                    <div className="muted">{title}</div>
                </div>
                <div className="duration">
                    约 <strong>{workout.estimatedMin}</strong> 分钟
                </div>
            </div>

            <button type="button" className="primary big" onClick={onStart}>
                ▶ 开始跟练
            </button>
            <div className="row">
                <button type="button" onClick={onRegenerate}>换一批动作</button>
                {!savedName && <button type="button" onClick={onSave}>保存为模板</button>}
                <button type="button" onClick={onBack}>修改条件</button>
                <button type="button" onClick={onOpenLibrary}>＋ 从动作库添加</button>
            </div>

            {workout.blocks.map((b, bi) => (
                <div key={bi} className="block">
                    <h3>{b.title}</h3>
                    <div className="muted small">{blockSubtitle(b)}</div>
                    <ol>
                        {b.items.map((it, i) => {
                            const ex = EXERCISE_BY_ID[it.exerciseId];
                            return (
                                <li key={i}>
                                    <div className="item-head">
                                        <span className="item-name">{ex.name}</span>
                                        <span className="item-dose">
                                            {dose(it, !!ex.perSide)}
                                            <button
                                                type="button"
                                                className="item-remove"
                                                aria-label={`移除 ${ex.name}`}
                                                title="从这次训练中移除"
                                                onClick={() => onRemoveItem(bi, i)}
                                            >
                                                ×
                                            </button>
                                        </span>
                                    </div>
                                    <details className="item-detail">
                                        <summary className="muted small">
                                            {EXERCISE_MUSCLES[ex.id].primary.map((m) => MUSCLE_DETAIL_LABEL[m]).join(' · ')} — {ex.cues[0]}
                                        </summary>
                                        <ExerciseBodyMap exerciseId={ex.id} stretch={ex.category === 'stretch'} />
                                        <ul className="cues-list small">
                                            {ex.cues.map((c) => (
                                                <li key={c}>{c}</li>
                                            ))}
                                        </ul>
                                    </details>
                                </li>
                            );
                        })}
                    </ol>
                </div>
            ))}
        </section>
    );
}
