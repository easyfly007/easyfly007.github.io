import { CLIP_VIEWS, clipName, type ClipView } from './engine/clips';

// 动作视频片段放在 src/videos/<动作 id>/<视角>.mp4（单侧动作为 <视角>-left.mp4 / <视角>-right.mp4），
// 构建时自动收录，不需要另外登记
const FILES = import.meta.glob('./videos/*/*.mp4', { eager: true, query: '?url', import: 'default' }) as Record<string, string>;

function urlOf(exerciseId: string, name: string): string | undefined {
    return FILES[`./videos/${exerciseId}/${name}.mp4`];
}

// 这个动作（这一侧）已有的视角及其地址，按 正面 → 侧面 → 全身 的顺序
export function clipUrls(exerciseId: string, side?: 'left' | 'right'): { view: ClipView; url: string }[] {
    const out: { view: ClipView; url: string }[] = [];
    for (const view of CLIP_VIEWS) {
        const url = urlOf(exerciseId, clipName(view, side));
        if (url)
            out.push({ view, url });
    }
    return out;
}
