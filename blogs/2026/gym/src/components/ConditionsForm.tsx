import type { ReactNode } from 'react';
import { DURATIONS, EQUIPMENT_LABEL, GOAL_LABEL, INTENSITY_LABEL, LIMITATION_LABEL } from '../labels';
import type { Conditions, Equipment, Goal, Intensity, Limitation } from '../types';

interface Props {
    value: Conditions;
    onChange: (c: Conditions) => void;
    onGenerate: () => void;
}

function toggle<T>(list: T[], item: T): T[] {
    return list.includes(item) ? list.filter((x) => x !== item) : [...list, item];
}

function Chip(props: { on: boolean; onClick: () => void; children: ReactNode }) {
    return (
        <button type="button" className={props.on ? 'chip on' : 'chip'} aria-pressed={props.on} onClick={props.onClick}>
            {props.children}
        </button>
    );
}

function Field(props: { label: string; hint?: string; children: ReactNode }) {
    return (
        <div className="field">
            <div className="field-label">
                {props.label}
                {props.hint && <span className="hint">{props.hint}</span>}
            </div>
            <div className="chips">{props.children}</div>
        </div>
    );
}

export function ConditionsForm({ value: c, onChange, onGenerate }: Props) {
    const set = (patch: Partial<Conditions>) => onChange({ ...c, ...patch });

    return (
        <section className="card">
            <h2>今天的条件</h2>

            <Field label="器械" hint="徒手动作始终可用">
                {(Object.keys(EQUIPMENT_LABEL) as Equipment[]).map((e) => (
                    <Chip key={e} on={c.equipment.includes(e)} onClick={() => set({ equipment: toggle(c.equipment, e) })}>
                        {EQUIPMENT_LABEL[e]}
                    </Chip>
                ))}
            </Field>

            <Field label="场地">
                <Chip on={c.location === 'indoor'} onClick={() => set({ location: 'indoor' })}>室内</Chip>
                <Chip on={c.location === 'outdoor'} onClick={() => set({ location: 'outdoor' })}>室外</Chip>
                <Chip on={c.space === 'large'} onClick={() => set({ space: c.space === 'large' ? 'small' : 'large' })}>
                    空间够跑动
                </Chip>
                <Chip on={c.quiet} onClick={() => set({ quiet: !c.quiet })}>需要安静</Chip>
            </Field>

            <Field label="装备与练后">
                <Chip on={c.hasShoes} onClick={() => set({ hasShoes: !c.hasShoes })}>有运动鞋</Chip>
                <Chip on={c.canShower} onClick={() => set({ canShower: !c.canShower })}>练完能洗澡</Chip>
            </Field>

            <Field label="时长">
                {DURATIONS.map((d) => (
                    <Chip key={d} on={c.durationMin === d} onClick={() => set({ durationMin: d })}>
                        {d} 分钟
                    </Chip>
                ))}
            </Field>

            <Field label="强度" hint={c.canShower ? undefined : '不能洗澡时会自动选低出汗动作'}>
                {([1, 2, 3] as Intensity[]).map((i) => (
                    <Chip key={i} on={c.intensity === i} onClick={() => set({ intensity: i })}>
                        {INTENSITY_LABEL[i]}
                    </Chip>
                ))}
            </Field>

            <Field label="目标">
                {(Object.keys(GOAL_LABEL) as Goal[]).map((g) => (
                    <Chip key={g} on={c.goal === g} onClick={() => set({ goal: g })}>
                        {GOAL_LABEL[g]}
                    </Chip>
                ))}
            </Field>

            <Field label="需要避开的部位" hint="可选">
                {(Object.keys(LIMITATION_LABEL) as Limitation[]).map((l) => (
                    <Chip key={l} on={c.limitations.includes(l)} onClick={() => set({ limitations: toggle(c.limitations, l) })}>
                        {LIMITATION_LABEL[l]}
                    </Chip>
                ))}
            </Field>

            <button type="button" className="primary big" onClick={onGenerate}>
                生成训练
            </button>
        </section>
    );
}
