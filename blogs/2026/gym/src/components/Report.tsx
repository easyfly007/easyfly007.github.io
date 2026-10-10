import { useState } from 'react';
import { summarize } from '../engine/report';
import { formatDuration, MUSCLE_LABEL, recordTitle, shortDateTime } from '../labels';
import type { WorkoutRecord } from '../types';
import { TrendCharts } from './History';

interface Props {
    record: WorkoutRecord;
    history: WorkoutRecord[];
    weightKg?: number;
    onWeightChange: (kg: number) => void;
    onBack: () => void;
}

export function Report({ record, history, weightKg, onWeightChange, onBack }: Props) {
    const s = summarize(record, weightKg);
    const [weightInput, setWeightInput] = useState('');
    const percent = Math.round(s.completion * 100);
    const full = percent >= 99;
    const maxMuscle = Math.max(1, ...s.muscles.map((m) => m.seconds));
    // 趋势图截至这次训练当天，打开旧记录时看到的是当时的情况
    const upTo = history.filter((r) => r.startedAt <= record.startedAt);

    function submitWeight() {
        const kg = Number(weightInput);
        if (kg >= 20 && kg <= 300)
            onWeightChange(kg);
    }

    return (
        <section className="card">
            <div className="row space">
                <h2>{full ? '🎉 训练完成' : `训练报告 · 完成 ${percent}%`}</h2>
                <button type="button" onClick={onBack}>返回</button>
            </div>
            <p className="muted small">
                {recordTitle(record)} · {shortDateTime(record.startedAt)}
            </p>

            <div className="stats">
                <div className="stat">
                    <div className="stat-label">训练时长</div>
                    <div className="stat-value">{formatDuration(Math.round(s.doneSec))}</div>
                    <div className="muted small">计划 {formatDuration(s.plannedSec)}</div>
                </div>
                <div className="stat">
                    <div className="stat-label">完成度</div>
                    <div className="stat-value">{percent}%</div>
                    <div className="muted small">组数 {s.setsDone} / {s.setsPlanned}</div>
                </div>
                <div className="stat">
                    <div className="stat-label">动作时间</div>
                    <div className="stat-value">{formatDuration(Math.round(s.workSec))}</div>
                    <div className="muted small">不含休息</div>
                </div>
                <div className="stat">
                    <div className="stat-label">消耗热量（估算）</div>
                    {s.kcal !== undefined ? (
                        <>
                            <div className="stat-value">约 {s.kcal} 千卡</div>
                            <div className="muted small">按体重 {weightKg} 公斤</div>
                        </>
                    ) : (
                        <div className="row">
                            <input
                                type="number"
                                inputMode="decimal"
                                placeholder="体重 kg"
                                aria-label="体重（公斤）"
                                value={weightInput}
                                onChange={(e) => setWeightInput(e.target.value)}
                            />
                            <button type="button" onClick={submitWeight}>计算</button>
                        </div>
                    )}
                </div>
            </div>

            {s.muscles.length > 0 && (
                <>
                    <h3 className="section-title">练到的部位</h3>
                    <ul className="meters">
                        {s.muscles.map((m) => (
                            <li key={m.muscle}>
                                <span>{MUSCLE_LABEL[m.muscle]}</span>
                                <span className="meter">
                                    <span style={{ width: `${(m.seconds / maxMuscle) * 100}%` }} />
                                </span>
                                <span className="muted small">{formatDuration(Math.round(m.seconds))}</span>
                            </li>
                        ))}
                    </ul>
                </>
            )}

            <h3 className="section-title">动作明细</h3>
            <ul className="report-list">
                {s.exercises.map((e) => (
                    <li key={e.exerciseId} className={e.setsDone === 0 ? 'muted' : undefined}>
                        <span>{e.name}</span>
                        <span className="small">
                            {e.setsDone} / {e.setsPlanned} 组
                            {e.reps !== undefined ? ` · ${e.reps} 次` : ` · ${Math.round(e.seconds)} 秒`}
                        </span>
                    </li>
                ))}
            </ul>

            <h3 className="section-title">最近的训练</h3>
            <TrendCharts records={upTo} today={new Date(record.startedAt)} />
        </section>
    );
}
