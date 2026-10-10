import { useState } from 'react';

export interface Bar {
    label: string;              // 横轴文字
    value: number;
    tip: string;                // 悬停/点按时显示的说明
}

interface Props {
    title: string;
    bars: Bar[];
    unit: string;               // 柱顶数值的单位，如"分"
    summary: string;            // 没有选中柱子时显示的总述
}

const W = 320;
const H = 150;
const TOP = 20;
const BOTTOM = 22;
const BAR_MAX = 24;
const RADIUS = 4;

// 顶端圆角、底端方角的柱子
function barPath(x: number, y: number, w: number, base: number): string {
    const r = Math.min(RADIUS, w / 2, base - y);
    return `M${x},${base} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${base} Z`;
}

// 单序列柱状图：柱子是热区，悬停或点按显示说明；有数值的柱子在顶端直接标注
export function BarChart({ title, bars, unit, summary }: Props) {
    const [active, setActive] = useState<number | null>(null);
    const max = Math.max(1, ...bars.map((b) => b.value));
    const base = H - BOTTOM;
    const slot = W / bars.length;
    const barW = Math.min(BAR_MAX, slot * 0.6);

    return (
        <figure className="chart">
            <figcaption>
                <span className="chart-title">{title}</span>
                <span className="muted small">{active === null ? summary : bars[active].tip}</span>
            </figcaption>
            <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}：${bars.map((b) => b.tip).join('；')}`}>
                <line x1={0} x2={W} y1={base + 0.5} y2={base + 0.5} className="chart-axis" />
                {bars.map((b, i) => {
                    const cx = slot * i + slot / 2;
                    const h = (b.value / max) * (base - TOP);
                    const y = base - h;
                    return (
                        <g
                            key={b.label + i}
                            className={active === i ? 'chart-bar on' : 'chart-bar'}
                            onMouseEnter={() => setActive(i)}
                            onMouseLeave={() => setActive(null)}
                            onClick={() => setActive(active === i ? null : i)}
                        >
                            <rect x={slot * i} y={0} width={slot} height={H} fill="transparent" />
                            {b.value > 0 && <path d={barPath(cx - barW / 2, y, barW, base)} />}
                            {b.value > 0 && (
                                <text x={cx} y={y - 5} textAnchor="middle" className="chart-value">
                                    {Math.round(b.value)}{unit}
                                </text>
                            )}
                            <text x={cx} y={H - 6} textAnchor="middle" className="chart-label">
                                {b.label}
                            </text>
                        </g>
                    );
                })}
            </svg>
        </figure>
    );
}
