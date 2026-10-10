import { CLIP_VIEWS, clipName, type ClipView } from './engine/clips';
import type { Gender } from './types';

// 动作视频片段放在 src/videos/<教练 female|male>/<动作 id>/<视角>.mp4
// （单侧动作为 <视角>-left.mp4 / <视角>-right.mp4），也接受 .webm（浏览器录制的片段）。
// 构建时自动收录，不需要另外登记
const FILES = import.meta.glob('./videos/*/*/*.{mp4,webm}', { eager: true, query: '?url', import: 'default' }) as Record<string, string>;

function urlOf(coach: Gender, exerciseId: string, name: string): string | undefined {
    const base = `./videos/${coach}/${exerciseId}/${name}`;
    return FILES[`${base}.mp4`] ?? FILES[`${base}.webm`];
}

function clipsOf(coach: Gender, exerciseId: string, side?: 'left' | 'right'): { view: ClipView; url: string }[] {
    const out: { view: ClipView; url: string }[] = [];
    for (const view of CLIP_VIEWS) {
        const url = urlOf(coach, exerciseId, clipName(view, side));
        if (url)
            out.push({ view, url });
    }
    return out;
}

// 这个动作（这一侧）已有的视角及其地址，按 正面 → 侧面 → 全身 的顺序。
// 所选教练还没有这个动作的片段时，用另一位教练的，总比没有画面好
export function clipUrls(exerciseId: string, side: 'left' | 'right' | undefined, coach: Gender): { view: ClipView; url: string }[] {
    const own = clipsOf(coach, exerciseId, side);
    if (own.length > 0)
        return own;
    return clipsOf(coach === 'female' ? 'male' : 'female', exerciseId, side);
}
