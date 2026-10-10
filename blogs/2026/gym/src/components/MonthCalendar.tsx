import { useState } from 'react';
import { dayKey, monthGrid, monthSummary } from '../engine/report';
import { MUSCLE_LABEL, recordTitle, shortDateTime } from '../labels';
import type { WorkoutRecord } from '../types';

interface Props {
    records: WorkoutRecord[];
    weightKg?: number;
    today: Date;
    onOpen: (r: WorkoutRecord) => void;
}

const WEEK = ['一', '二', '三', '四', '五', '六', '日'];

// 训练分钟数分 3 档着色（同一色相由浅到深），没练的日子不着色
function level(seconds: number): number {
    if (seconds <= 0)
        return 0;
    if (seconds < 15 * 60)
        return 1;
    if (seconds < 30 * 60)
        return 2;
    return 3;
}

// 一个月的训练日历 + 本月总结；点某天看当天的训练
export function MonthCalendar({ records, weightKg, today, onOpen }: Props) {
    const [ym, setYm] = useState({ y: today.getFullYear(), m: today.getMonth() });
    const [picked, setPicked] = useState<string | null>(null);
    const weeks = monthGrid(ym.y, ym.m);
    const sum = monthSummary(records, ym.y, ym.m, weightKg);
    const todayKey = dayKey(today);
    const pickedDay = picked ? sum.byDay.get(picked) : undefined;

    function shift(delta: number) {
        const d = new Date(ym.y, ym.m + delta, 1);
        setYm({ y: d.getFullYear(), m: d.getMonth() });
        setPicked(null);
    }

    return (
        <section className="card">
            <div className="row space">
                <h2>训练日历</h2>
                <div className="row">
                    <button type="button" aria-label="上个月" onClick={() => shift(-1)}>‹</button>
                    <span className="month-title">{ym.y} 年 {ym.m + 1} 月</span>
                    <button type="button" aria-label="下个月" onClick={() => shift(1)}>›</button>
                </div>
            </div>

            <table className="calendar">
                <thead>
                    <tr>
                        {WEEK.map((w) => (
                            <th key={w}>{w}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {weeks.map((week, wi) => (
                        <tr key={wi}>
                            {week.map((d, di) => {
                                if (!d)
                                    return <td key={di} />;
                                const key = dayKey(d);
                                const day = sum.byDay.get(key);
                                const lv = level(day?.seconds ?? 0);
                                const cls = [`cal-day`, `lv${lv}`, key === todayKey ? 'today' : '', key === picked ? 'picked' : ''].join(' ');
                                const label = day ? `${d.getMonth() + 1}月${d.getDate()}日：${day.records.length} 次，${Math.round(day.seconds / 60)} 分钟` : `${d.getMonth() + 1}月${d.getDate()}日：没有训练`;
                                return (
                                    <td key={di}>
                                        <button type="button" className={cls} aria-label={label} title={label} onClick={() => setPicked(key === picked ? null : key)}>
                                            <span>{d.getDate()}</span>
                                            {day && <span className="cal-min">{Math.round(day.seconds / 60)}′</span>}
                                        </button>
                                    </td>
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
            <div className="cal-legend muted small">
                <span><i className="cal-key lv1" />少于 15 分钟</span>
                <span><i className="cal-key lv2" />15–30 分钟</span>
                <span><i className="cal-key lv3" />30 分钟以上</span>
            </div>

            {pickedDay && (
                <ul className="saved">
                    {pickedDay.records.map((r) => (
                        <li key={r.id}>
                            <button type="button" className="saved-main" onClick={() => onOpen(r)}>
                                <span className="item-name">{recordTitle(r)}</span>
                                <span className="muted small">{shortDateTime(r.startedAt)}</span>
                            </button>
                        </li>
                    ))}
                </ul>
            )}

            <h3 className="section-title">本月总结</h3>
            {sum.count === 0 ? (
                <p className="muted">这个月还没有训练记录。</p>
            ) : (
                <>
                    <div className="stats">
                        <div className="stat">
                            <div className="stat-label">训练次数</div>
                            <div className="stat-value">{sum.count} 次</div>
                        </div>
                        <div className="stat">
                            <div className="stat-label">训练天数</div>
                            <div className="stat-value">{sum.activeDays} 天</div>
                        </div>
                        <div className="stat">
                            <div className="stat-label">总时长</div>
                            <div className="stat-value">{Math.round(sum.seconds / 60)} 分钟</div>
                        </div>
                        <div className="stat">
                            <div className="stat-label">消耗热量（估算）</div>
                            <div className="stat-value">{sum.kcal !== undefined ? `约 ${Math.round(sum.kcal)} 千卡` : '—'}</div>
                        </div>
                    </div>
                    <p className="small">
                        练得最多的部位：{sum.muscles.slice(0, 3).map((m) => MUSCLE_LABEL[m.muscle]).join('、')}
                    </p>
                </>
            )}
        </section>
    );
}
