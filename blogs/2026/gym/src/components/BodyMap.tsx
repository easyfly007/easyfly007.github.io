import { EXERCISE_MUSCLES, MUSCLE_DETAIL_LABEL, type MuscleDetail } from '../data/muscles';

// 人体肌肉图：正面 + 背面。只画人物轮廓和用到的肌肉：主要肌肉橙色，次要肌肉浅橙色，没用到的不画。
// 每个视图的坐标系是 100 × 230，人物居中（x = 50），体型按偏健壮的正常人；
// 左右对称的形状只写一侧（画面左半边），另一侧镜像
interface Shape {
    muscle?: MuscleDetail;
    d: string;
    mirror?: boolean;           // 是否再镜像画到另一侧
}

// 人物轮廓：头、颈，加上从中线出发绕半个身体一圈的闭合路径（含手臂、腿、手脚）
const BODY: Shape[] = [
    { d: 'M50,3.5 a9.5,11.5 0 1,0 0.01,0 Z' },
    { d: 'M44.5,22 L55.5,22 L56,30 L44,30 Z' },
    {
        d: 'M50.5,26 L45,26 C45,29 44,31 41,32 C35,33 28,34 24,38 C20,41 19,47 19,52 C18,58 17,64 17,70 '
            + 'C16,74 15,78 14,84 C13,90 13,96 13,100 C13,103 14,106 15,108 C14,112 15,117 18,118 '
            + 'C21,118 22,113 22,108 C23,104 23,98 24,92 C25,86 26,80 26,76 C27,70 28,64 29,58 '
            + 'C30.5,62 32,70 33,76 C34,82 33,88 31,94 C29,100 29,106 30,112 C30,124 31,134 33,146 '
            + 'C34,152 34,156 33,160 C31,168 31,178 33,190 C34,198 35,204 36,208 C33,211 32,214 35,215 '
            + 'L45,215 C46,213 46,210 45,207 C45,198 46,188 46,178 C46,168 47,160 47,154 '
            + 'C48,140 49,124 49.5,112 L50.5,104 Z',
        mirror: true,
    },
];

const FRONT: Shape[] = [
    { muscle: 'traps', d: 'M45,28.5 C43,31 40,32.2 37,33 L45,33.6 Z', mirror: true },
    { muscle: 'delts', d: 'M38,34.5 C31,33.5 24,36 21.5,43 C20.5,48 21,52 23,54 C26,48 30,42 38,38.5 Z', mirror: true },
    { muscle: 'chest', d: 'M49.5,36.5 C42,35.5 36.5,37.5 33.5,42 C32.5,48 34,53 38,55 C43,57 47,56 49.5,54 Z', mirror: true },
    { muscle: 'biceps', d: 'M24.5,56 C21.5,58.5 20.5,64 20.5,70 C21.5,74 24,75 26,73 C27,67 28,61 28.2,56.5 Z', mirror: true },
    { muscle: 'forearms', d: 'M17.5,80 C15.5,86 15.5,94 16.2,101 L19.2,101 C21,94 23,86 24,80.5 Z', mirror: true },
    { muscle: 'abs', d: 'M44.5,58.5 h4.8 v7 h-4.8 Z', mirror: true },
    { muscle: 'abs', d: 'M44.5,67 h4.8 v7 h-4.8 Z', mirror: true },
    { muscle: 'abs', d: 'M44.5,75.5 h4.8 v7.5 h-4.8 Z', mirror: true },
    { muscle: 'abs', d: 'M45,84.5 h4.3 v7 h-4.3 Z', mirror: true },
    { muscle: 'obliques', d: 'M42.5,58 C38,60 35.5,66 35.5,74 C35.5,80 37.5,86 41,90 L43,90 C43,80 43,68 42.5,58 Z', mirror: true },
    { muscle: 'hip_flexors', d: 'M41,93.5 C38,96 37,101 38,106 L45,104 L48.5,97 Z', mirror: true },
    { muscle: 'quads', d: 'M33.5,109 C31.5,119 32.5,132 36,144 C39,148 43,148 45,145 C47,135 47,122 46,110.5 C43,105 37.5,104 33.5,109 Z', mirror: true },
    { muscle: 'adductors', d: 'M48.6,105 C47.2,112 46.6,124 46.8,134 L49.2,128 L49.6,106 Z', mirror: true },
    { muscle: 'calves', d: 'M34,162 C32.5,170 32.5,180 34.5,188 C36.5,191 38.5,190 39.5,187 C40.5,178 40.5,168 39.5,161 Z', mirror: true },
];

const BACK: Shape[] = [
    { muscle: 'lats', d: 'M47,47 C41,44 35,44 31.5,47.5 C31.5,58 33.5,68 37,77 C41,83 46,86 49.5,86 L49.5,62 Z', mirror: true },
    { muscle: 'traps', d: 'M50,25.5 C46,29 41,31 37,33 C41,37 45,44 50,60 C55,44 59,37 63,33 C59,31 54,29 50,25.5 Z' },
    { muscle: 'delts', d: 'M38,34.5 C31,33.5 24,36 21.5,43 C20.5,48 21,52 23,54 C26,48 30,42 38,38.5 Z', mirror: true },
    { muscle: 'triceps', d: 'M24.5,56 C21.5,58.5 20.5,64 20.5,70 C21.5,74 24,75 26,73 C27,67 28,61 28.2,56.5 Z', mirror: true },
    { muscle: 'forearms', d: 'M17.5,80 C15.5,86 15.5,94 16.2,101 L19.2,101 C21,94 23,86 24,80.5 Z', mirror: true },
    { muscle: 'lower_back', d: 'M44.5,72 C43.5,78 43.5,86 44.5,94 L49.5,94 L49.5,72 Z', mirror: true },
    { muscle: 'glutes', d: 'M49.7,96 C42,94 35,96 32,104 C31,112 35,118 41,119 C45,119 48,117 49.7,114 Z', mirror: true },
    { muscle: 'hamstrings', d: 'M33.5,121 C32.5,131 34,141 37,149 C40,152 44,152 46,148 C47.5,138 47.5,128 47,121 C42,118 37,118 33.5,121 Z', mirror: true },
    { muscle: 'calves', d: 'M33.5,160 C31,168 31,178 34,186 C37,190 41,190 43,186 C45,178 45.5,168 44.5,160 C41,156 36,156 33.5,160 Z', mirror: true },
];

const MIRROR = 'matrix(-1 0 0 1 100 0)';

function paths(shapes: Shape[], cls: string) {
    return shapes.map((s, i) => (
        <g key={i}>
            <path d={s.d} className={cls} />
            {s.mirror && <path d={s.d} className={cls} transform={MIRROR} />}
        </g>
    ));
}

function Figure({ shapes, primary, secondary, dx, label }: {
    shapes: Shape[];
    primary: MuscleDetail[];
    secondary: MuscleDetail[];
    dx: number;
    label: string;
}) {
    // 左右两半的轮廓各自闭合，先整体描边、再整体填充，中线处的描边会被填充盖住，只留下外轮廓
    const used = (level: MuscleDetail[]) => shapes.filter((s) => s.muscle && level.includes(s.muscle));
    return (
        <g transform={`translate(${dx} 0)`}>
            {paths(BODY, 'bm-outline')}
            {paths(BODY, 'bm-body')}
            {paths(used(secondary), 'bm-secondary')}
            {paths(used(primary), 'bm-primary')}
            <text x={50} y={227} textAnchor="middle" className="bm-label">{label}</text>
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
