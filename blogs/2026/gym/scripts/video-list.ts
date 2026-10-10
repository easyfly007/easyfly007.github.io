// 生成动作视频片段清单 docs/video-clips.csv：每个动作 × 每个视角（单侧动作再 × 左右）一行，含文件路径和 AI 生成提示词。
// 用法（在本目录下）：npx vite-node scripts/video-list.ts
import { writeFileSync } from 'node:fs';
import { EXERCISES } from '../src/data/exercises';
import { CLIP_VIEW_LABEL, CLIP_VIEWS, clipName, cycleSeconds, TIME_CLIP_SEC, type ClipView } from '../src/engine/clips';
import { EQUIPMENT_LABEL } from '../src/labels';
import type { Exercise } from '../src/types';

const CLIP_SEC = 5;

const VIEW_DESC: Record<ClipView, string> = {
    front: '机位在人物正前方，与腰部同高平视，全身完整入镜',
    side: '机位在人物正侧面（人物左侧对着镜头），与腰部同高平视，全身完整入镜',
    full: '机位在人物右前方 45 度、略高于头部向下俯拍，全身和地面完整入镜',
};

const SIDE_LABEL = { left: '左侧', right: '右侧' };

function prompt(e: Exercise, view: ClipView, side?: 'left' | 'right'): string {
    const parts = [
        '同一位健身教练（使用角色参考图，服装、发型、背景与其他片段完全一致），干净的浅色室内背景',
        VIEW_DESC[view],
        '固定机位，镜头不移动、不变焦、不剪辑，无字幕、无水印、无音乐',
    ];
    if (e.equipment.length > 0)
        parts.push(`使用${e.equipment.map((q) => EQUIPMENT_LABEL[q]).join('和')}`);
    const name = side ? `${e.name}（${SIDE_LABEL[side]}）` : e.name;
    const cues = e.cues.join('；');
    if (e.mode === 'reps') {
        parts.push(`从标准起始姿态开始，完整做一次${name}，动作要点：${cues}`);
        parts.push(`只做一次，动作匀速、标准，正好占满 ${CLIP_SEC} 秒`);
    } else {
        parts.push(`持续做${name}，动作要点：${cues}`);
        parts.push(`节奏均匀，正好 ${CLIP_SEC} 秒`);
    }
    parts.push('最后一帧回到与第一帧完全相同的姿态，便于和下一段无缝衔接');
    return parts.join('。') + '。';
}

function csvCell(s: string | number): string {
    const t = String(s);
    return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
}

const header = ['文件', '动作 id', '动作', '视角', '侧', '计时/计次', '片段时长(秒)', '跟练中每段时长(秒)', '提示词'];
const rows: (string | number)[][] = [];
for (const e of EXERCISES) {
    const sides: ('left' | 'right' | undefined)[] = e.perSide ? ['left', 'right'] : [undefined];
    for (const side of sides) {
        for (const view of CLIP_VIEWS) {
            rows.push([
                `src/videos/${e.id}/${clipName(view, side)}.mp4`,
                e.id,
                e.name,
                CLIP_VIEW_LABEL[view],
                side ? SIDE_LABEL[side] : '',
                e.mode === 'reps' ? '计次' : '计时',
                CLIP_SEC,
                e.mode === 'reps' ? cycleSeconds(e) : TIME_CLIP_SEC,
                prompt(e, view, side),
            ]);
        }
    }
}

const out = new URL('../docs/video-clips.csv', import.meta.url);
// 带 BOM，Excel 打开中文不乱码
writeFileSync(out, '﻿' + [header, ...rows].map((r) => r.map(csvCell).join(',')).join('\n') + '\n');
console.log(`${rows.length} 个片段 → docs/video-clips.csv`);
