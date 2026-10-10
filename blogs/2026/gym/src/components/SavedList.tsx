import type { SavedWorkout } from '../types';

interface Props {
    items: SavedWorkout[];
    onOpen: (s: SavedWorkout) => void;
    onStart: (s: SavedWorkout) => void;
    onRegenerate: (s: SavedWorkout) => void;
    onDelete: (s: SavedWorkout) => void;
}

function shortDate(iso?: string): string {
    if (!iso)
        return '未练过';
    const d = new Date(iso);
    return `${d.getMonth() + 1}月${d.getDate()}日`;
}

export function SavedList({ items, onOpen, onStart, onRegenerate, onDelete }: Props) {
    return (
        <section className="card">
            <h2>我的模板</h2>
            {items.length === 0 && <p className="muted">还没有保存的训练。生成一次训练后点"保存为模板"，下次就能直接开练。</p>}
            <ul className="saved">
                {items.map((s) => (
                    <li key={s.id}>
                        <button type="button" className="saved-main" onClick={() => onOpen(s)}>
                            <span className="item-name">{s.name}</span>
                            <span className="muted small">
                                约 {s.workout.estimatedMin} 分钟 · 上次：{shortDate(s.lastUsedAt)}
                            </span>
                        </button>
                        <div className="row">
                            <button type="button" className="primary" onClick={() => onStart(s)}>▶ 开练</button>
                            <button type="button" onClick={() => onRegenerate(s)} title="同样的条件，换一批动作">换一批</button>
                            <button type="button" className="danger" onClick={() => onDelete(s)}>删除</button>
                        </div>
                    </li>
                ))}
            </ul>
        </section>
    );
}
