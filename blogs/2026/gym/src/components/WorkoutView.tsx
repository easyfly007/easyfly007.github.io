import { EXERCISE_BY_ID } from '../data/exercises';
import { MUSCLE_LABEL } from '../labels';
import type { Block, Workout, WorkoutItem } from '../types';

interface Props {
    workout: Workout;
    title: string;
    savedName?: string;
    onStart: () => void;
    onRegenerate: () => void;
    onSave: () => void;
    onBack: () => void;
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

export function WorkoutView({ workout, title, savedName, onStart, onRegenerate, onSave, onBack }: Props) {
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
                                        <span className="item-dose">{dose(it, !!ex.perSide)}</span>
                                    </div>
                                    <div className="muted small">
                                        {ex.muscles.map((m) => MUSCLE_LABEL[m]).join(' · ')} — {ex.cues[0]}
                                    </div>
                                </li>
                            );
                        })}
                    </ol>
                </div>
            ))}
        </section>
    );
}
