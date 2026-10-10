import { dailyTotals, playedSeconds, streakDays, weeklyTotals } from '../engine/report';
import { recordTitle, shortDateTime } from '../labels';
import type { WorkoutRecord } from '../types';
import { BarChart } from './BarChart';

const WEEKDAY = ['日', '一', '二', '三', '四', '五', '六'];

function minutes(sec: number): number {
    return Math.round(sec / 60);
}

// 最近 7 天每天的训练分钟数 + 最近 8 周的趋势
export function TrendCharts({ records, today }: { records: WorkoutRecord[]; today: Date }) {
    const days = dailyTotals(records, today);
    const weeks = weeklyTotals(records, today);
    const weekSec = days.reduce((s, d) => s + d.seconds, 0);
    const weekCount = days.reduce((s, d) => s + d.count, 0);
    const activeDays = days.filter((d) => d.count > 0).length;
    const streak = streakDays(records, today);

    return (
        <>
            <div className="stats">
                <div className="stat">
                    <div className="stat-label">近 7 天训练</div>
                    <div className="stat-value">{weekCount} 次</div>
                </div>
                <div className="stat">
                    <div className="stat-label">近 7 天时长</div>
                    <div className="stat-value">{minutes(weekSec)} 分钟</div>
                </div>
                <div className="stat">
                    <div className="stat-label">练了几天</div>
                    <div className="stat-value">{activeDays} / 7</div>
                </div>
                <div className="stat">
                    <div className="stat-label">连续打卡</div>
                    <div className="stat-value">{streak} 天</div>
                </div>
            </div>
            <BarChart
                title="最近 7 天"
                unit="分"
                summary="每天训练分钟数，点柱子看详情"
                bars={days.map((d, i) => ({
                    label: i === days.length - 1 ? '今天' : `周${WEEKDAY[d.date.getDay()]}`,
                    value: d.seconds / 60,
                    tip: `${d.date.getMonth() + 1}月${d.date.getDate()}日：${d.count} 次，${minutes(d.seconds)} 分钟`,
                }))}
            />
            <BarChart
                title="最近 8 周趋势"
                unit="分"
                summary="每周训练分钟数（周一开始）"
                bars={weeks.map((w, i) => ({
                    label: i === weeks.length - 1 ? '本周' : `${w.date.getMonth() + 1}/${w.date.getDate()}`,
                    value: w.seconds / 60,
                    tip: `${w.date.getMonth() + 1}月${w.date.getDate()}日那周：${w.count} 次，${minutes(w.seconds)} 分钟`,
                }))}
            />
        </>
    );
}

interface Props {
    records: WorkoutRecord[];
    onOpen: (r: WorkoutRecord) => void;
}

// 首页的"训练记录"卡片
export function History({ records, onOpen }: Props) {
    if (records.length === 0) {
        return (
            <section className="card">
                <h2>训练记录</h2>
                <p className="muted">还没有训练记录。跟练结束后会自动生成训练报告并记在这里。</p>
            </section>
        );
    }
    return (
        <section className="card">
            <h2>训练记录</h2>
            <TrendCharts records={records} today={new Date()} />
            <ul className="saved">
                {records.slice(0, 10).map((r) => (
                    <li key={r.id}>
                        <button type="button" className="saved-main" onClick={() => onOpen(r)}>
                            <span className="item-name">{recordTitle(r)}</span>
                            <span className="muted small">
                                {shortDateTime(r.startedAt)} · {minutes(playedSeconds(r))} 分钟
                            </span>
                        </button>
                    </li>
                ))}
            </ul>
        </section>
    );
}
