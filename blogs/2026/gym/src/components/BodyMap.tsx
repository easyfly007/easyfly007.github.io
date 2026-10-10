import { EXERCISE_MUSCLES, MUSCLE_DETAIL_LABEL, type MuscleDetail } from '../data/muscles';

// 人体肌肉图：正面 + 背面。主要肌肉橙色，次要肌肉浅橙色，其余灰色。
// 每个视图的坐标系是 100 × 230，人物居中（x = 50）；左右对称的肌肉只画一侧，另一侧镜像
interface Shape {
    muscle?: MuscleDetail;      // 不填则是身体底色（头、关节、手脚等）
    d: string;
    mirror?: boolean;           // 是否再镜像画到另一侧
}

const BODY: Shape[] = [
    { d: 'M50,5 a9,11 0 1,0 0.01,0 Z' },
    { d: 'M45,24 h10 v9 h-10 Z' },
    { d: 'M37,34 Q50,30 63,34 L66,44 Q64,60 61,80 L63,96 L37,96 L39,80 Q36,60 34,44 Z' },
    { d: 'M34,36 Q27,38 26,48 L24,68 Q26,72 30,70 L33,50 Z', mirror: true },
    { d: 'M24,70 Q20,80 19,96 L23,97 Q27,84 29,72 Z', mirror: true },
    { d: 'M20,96 a3.5,5 0 1,0 0.01,0 Z', mirror: true },
    { d: 'M37,95 Q35,120 38,148 L48,148 Q50,120 50,97 Z', mirror: true },
    { d: 'M38,148 Q36,175 39,203 L45,203 Q48,175 48,148 Z', mirror: true },
    { d: 'M38,203 L46,203 L47,211 Q41,213 36,211 Z', mirror: true },
];

const FRONT: Shape[] = [
    { muscle: 'traps', d: 'M45,29 Q41,32 38,34 L45.5,34 Z', mirror: true },
    { muscle: 'delts', d: 'M36,35 Q28,36 27,44 Q28,49 31,50 Q34,44 39,39 Z', mirror: true },
    { muscle: 'chest', d: 'M49,37 Q41,35 38,39 Q35,46 38,51 Q44,54 49,52 Z', mirror: true },
    { muscle: 'biceps', d: 'M30,51 Q27,52 26,58 L25,66 Q27,68 29,67 Q31,60 32,52 Z', mirror: true },
    { muscle: 'forearms', d: 'M25,71 Q21,78 20,92 L23,93 Q26,82 28,73 Z', mirror: true },
    { muscle: 'abs', d: 'M44.5,55 h4.5 v7.5 h-4.5 Z', mirror: true },
    { muscle: 'abs', d: 'M44.5,64 h4.5 v7.5 h-4.5 Z', mirror: true },
    { muscle: 'abs', d: 'M44.5,73 h4.5 v8 h-4.5 Z', mirror: true },
    { muscle: 'obliques', d: 'M43,55 Q38,58 38,68 Q38,78 41,85 L43,85 Z', mirror: true },
    { muscle: 'hip_flexors', d: 'M43,88 Q39,90 38.5,96 L44,99 L48.5,92 Z', mirror: true },
    { muscle: 'quads', d: 'M38,100 Q36,118 39,143 Q43,147 46,143 Q47,124 47,104 Q44,100 40,99 Z', mirror: true },
    { muscle: 'adductors', d: 'M48.5,100 Q47.5,114 47.5,128 L49.5,126 L49.5,100 Z', mirror: true },
    { muscle: 'calves', d: 'M38.5,155 Q36,168 38.5,183 Q41,186 43,183 Q44.5,168 43.5,155 Q41,152 38.5,155 Z', mirror: true },
];

const BACK: Shape[] = [
    { muscle: 'lats', d: 'M48,46 Q41,43 36.5,44 Q36,58 40,70 Q45,77 49,77 Z', mirror: true },
    { muscle: 'traps', d: 'M50,26 L41,33 Q44,40 50,57 Q56,40 59,33 Z' },
    { muscle: 'delts', d: 'M36,35 Q28,36 27,44 Q28,49 31,50 Q34,44 39,39 Z', mirror: true },
    { muscle: 'triceps', d: 'M30,51 Q27,52 26,58 L25,66 Q27,68 29,67 Q31,60 32,52 Z', mirror: true },
    { muscle: 'forearms', d: 'M25,71 Q21,78 20,92 L23,93 Q26,82 28,73 Z', mirror: true },
    { muscle: 'lower_back', d: 'M44.5,73 h5 v17 h-5 Z', mirror: true },
    { muscle: 'glutes', d: 'M49.5,93 Q41,90 37.5,96 Q36.5,106 40,110 Q46,112 49.5,108 Z', mirror: true },
    { muscle: 'hamstrings', d: 'M38.5,114 Q37.5,128 39.5,144 Q43,147 46.5,144 Q48,128 48,114 Q43,111.5 38.5,114 Z', mirror: true },
    { muscle: 'calves', d: 'M38.5,154 Q35.5,166 38.5,180 Q41.5,184 44.5,180 Q47,166 46,154 Q42,151 38.5,154 Z', mirror: true },
];

const MIRROR = 'matrix(-1 0 0 1 100 0)';

type Level = 'primary' | 'secondary' | 'none';

function levelOf(muscle: MuscleDetail | undefined, primary: MuscleDetail[], secondary: MuscleDetail[]): Level | 'base' {
    if (!muscle)
        return 'base';
    if (primary.includes(muscle))
        return 'primary';
    if (secondary.includes(muscle))
        return 'secondary';
    return 'none';
}

function Figure({ shapes, primary, secondary, dx, label }: {
    shapes: Shape[];
    primary: MuscleDetail[];
    secondary: MuscleDetail[];
    dx: number;
    label: string;
}) {
    const all = [...BODY, ...shapes];
    return (
        <g transform={`translate(${dx} 0)`}>
            {all.map((s, i) => {
                const cls = `bm-${levelOf(s.muscle, primary, secondary)}`;
                const title = s.muscle ? <title>{MUSCLE_DETAIL_LABEL[s.muscle]}</title> : null;
                return (
                    <g key={i}>
                        <path d={s.d} className={cls}>{title}</path>
                        {s.mirror && <path d={s.d} className={cls} transform={MIRROR}>{title}</path>}
                    </g>
                );
            })}
            <text x={50} y={226} textAnchor="middle" className="bm-label">{label}</text>
        </g>
    );
}

interface Props {
    primary: MuscleDetail[];
    secondary: MuscleDetail[];
    stretch?: boolean;          // 拉伸类动作：文字说"拉伸到"而不是"锻炼到"
    compact?: boolean;          // 只画图，不显示图例和肌肉名
}

export function BodyMap({ primary, secondary, stretch, compact }: Props) {
    const verb = stretch ? '拉伸' : '锻炼';
    const names = (list: MuscleDetail[]) => list.map((x) => MUSCLE_DETAIL_LABEL[x]).join('、');
    return (
        <figure className={compact ? 'body-map compact' : 'body-map'}>
            <svg viewBox="0 0 200 230" role="img" aria-label={`主要${verb}：${names(primary)}${secondary.length ? `；次要：${names(secondary)}` : ''}`}>
                <Figure shapes={FRONT} primary={primary} secondary={secondary} dx={0} label="正面" />
                <Figure shapes={BACK} primary={primary} secondary={secondary} dx={100} label="背面" />
            </svg>
            {!compact && (
                <figcaption>
                    <div>
                        <span className="bm-key bm-key-primary" />主要{verb}：{names(primary)}
                    </div>
                    {secondary.length > 0 && (
                        <div className="muted">
                            <span className="bm-key bm-key-secondary" />次要：{names(secondary)}
                        </div>
                    )}
                </figcaption>
            )}
        </figure>
    );
}

// 按动作 id 画肌肉图
export function ExerciseBodyMap({ exerciseId, stretch, compact }: { exerciseId: string; stretch?: boolean; compact?: boolean }) {
    const mm = EXERCISE_MUSCLES[exerciseId];
    if (!mm)
        return null;
    return <BodyMap primary={mm.primary} secondary={mm.secondary} stretch={stretch} compact={compact} />;
}
