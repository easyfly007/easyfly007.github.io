import { useEffect, useState } from 'react';
import { onVoicesChanged, setVoiceGender, speak, unlockAudio, voiceAvailable } from '../speech';
import type { Gender, Settings } from '../types';

interface Props {
    value: Settings;
    onChange: (s: Settings) => void;
}

const GENDER_LABEL: Record<Gender, string> = { female: '女', male: '男' };

export function SettingsCard({ value, onChange }: Props) {
    // 语音列表是异步加载的，加载完再判断本机有没有所选的声音
    const [, setVoicesVersion] = useState(0);
    useEffect(() => onVoicesChanged(() => setVoicesVersion((n) => n + 1)), []);
    const hasVoice = voiceAvailable(value.voice);

    function preview(voice: Gender) {
        unlockAudio();
        setVoiceGender(voice);
        speak('准备开始，第一个动作：徒手深蹲');
    }

    return (
        <section className="card">
            <h2>教练与语音</h2>
            <div className="field">
                <div className="field-label">
                    视频教练
                    <span className="hint">动作视频里示范的人</span>
                </div>
                <div className="chips">
                    {(['female', 'male'] as Gender[]).map((g) => (
                        <button
                            key={g}
                            type="button"
                            className={value.coach === g ? 'chip on' : 'chip'}
                            aria-pressed={value.coach === g}
                            onClick={() => onChange({ ...value, coach: g })}
                        >
                            {GENDER_LABEL[g]}教练
                        </button>
                    ))}
                </div>
            </div>
            <div className="field">
                <div className="field-label">
                    语音播报
                    {!hasVoice && (
                        <span className="hint">本机没有中文{GENDER_LABEL[value.voice]}声语音包，暂用调整音调代替</span>
                    )}
                </div>
                <div className="chips">
                    {(['female', 'male'] as Gender[]).map((g) => (
                        <button
                            key={g}
                            type="button"
                            className={value.voice === g ? 'chip on' : 'chip'}
                            aria-pressed={value.voice === g}
                            onClick={() => onChange({ ...value, voice: g })}
                        >
                            {GENDER_LABEL[g]}声
                        </button>
                    ))}
                    <button type="button" className="chip" onClick={() => preview(value.voice)}>
                        ▶ 试听
                    </button>
                </div>
            </div>
        </section>
    );
}
