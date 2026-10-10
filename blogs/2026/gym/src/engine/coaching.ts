import { EXERCISE_BY_ID } from '../data/exercises';
import type { Step } from '../types';

// 跟练中的口头提示：动作要点（短句）和进度提示（还剩几次/几秒）交替播报。
// 同一个动作的要点整节课只讲一遍（第二组起只报进度）；最后几秒留给倒计时提示音

export interface CoachEvent {
    at: number;                 // 在这一步开始后的第几秒播报
    text: string;
    kind: 'tip' | 'progress';
}

const FIRST_SEC = 5;            // 开头先播"开始，xx 几次"，留出时间
const GAP_SEC = 5;
const QUIET_END_SEC = 4;        // 最后 4 秒不插话（有倒计时提示音）

// 动作要点拆成短句
export function tipsOf(exerciseId: string): string[] {
    return EXERCISE_BY_ID[exerciseId].cues.flatMap((c) => c.split(/[，；。,;]/).map((t) => t.trim()).filter(Boolean));
}

function progressText(step: Step, at: number): string | null {
    const ex = EXERCISE_BY_ID[step.exerciseId];
    if (step.reps !== undefined) {
        const left = step.reps - Math.floor(at / ex.secPerRep);
        return left > 0 ? `还剩 ${left} 次` : null;
    }
    const left = Math.round(step.seconds - at);
    return left > 0 ? `还剩 ${left} 秒` : null;
}

// 整节课每一步的提示（下标与 steps 对应；非动作步骤为空）
export function buildCoaching(steps: Step[]): CoachEvent[][] {
    const used = new Map<string, number>();
    return steps.map((step) => {
        if (step.kind !== 'work')
            return [];
        const tips = tipsOf(step.exerciseId);
        const events: CoachEvent[] = [];
        let tipTurn = true;
        for (let at = FIRST_SEC; at <= step.seconds - QUIET_END_SEC; at += GAP_SEC) {
            const next = used.get(step.exerciseId) ?? 0;
            if (tipTurn && next < tips.length) {
                events.push({ at, text: tips[next], kind: 'tip' });
                used.set(step.exerciseId, next + 1);
            } else {
                const text = progressText(step, at);
                if (text)
                    events.push({ at, text, kind: 'progress' });
            }
            tipTurn = !tipTurn;
        }
        return events;
    });
}
