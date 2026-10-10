// 语音播报与提示音。浏览器不支持时静默跳过，不影响计时
import type { Gender } from './types';

// 常见系统自带的中文语音（Windows / macOS / iOS / Edge 在线语音），按名字判断男女声
const FEMALE_HINTS = ['xiaoxiao', 'xiaoyi', 'xiaohan', 'xiaomo', 'xiaorui', 'xiaoxuan', 'xiaoyan', 'huihui', 'yaoyao',
    'tingting', 'ting-ting', 'meijia', 'mei-jia', 'sinji', 'sin-ji', 'yu-shu', 'yushu', 'lili', 'shanshan',
    'google 普通话', 'female', '女'];
const MALE_HINTS = ['yunxi', 'yunjian', 'yunyang', 'yunye', 'yunhao', 'yunfeng', 'yunze', 'kangkang', 'li-mu', 'limu',
    'male', '男'];

export function voiceGender(v: Pick<SpeechSynthesisVoice, 'name'>): Gender | null {
    const name = v.name.toLowerCase();
    // 先判断男声：'female' 里含有 'male'
    if (MALE_HINTS.some((h) => name.includes(h)) && !name.includes('female'))
        return 'male';
    if (FEMALE_HINTS.some((h) => name.includes(h)))
        return 'female';
    return null;
}

export interface VoiceChoice {
    voice: SpeechSynthesisVoice | null;
    matched: boolean;           // 是否找到了所选性别的中文语音
}

// 在中文语音里选所选性别的一个（优先普通话 zh-CN）；找不到就退回任意中文语音
export function pickVoice(voices: SpeechSynthesisVoice[], gender: Gender): VoiceChoice {
    const zh = voices.filter((v) => v.lang.toLowerCase().replace('_', '-').startsWith('zh'));
    const rank = (v: SpeechSynthesisVoice) => (v.lang.toLowerCase().replace('_', '-') === 'zh-cn' ? 0 : 1);
    zh.sort((a, b) => rank(a) - rank(b));
    const same = zh.find((v) => voiceGender(v) === gender);
    if (same)
        return { voice: same, matched: true };
    return { voice: zh[0] ?? null, matched: false };
}

let wantGender: Gender = 'female';

function availableVoices(): SpeechSynthesisVoice[] {
    if (!('speechSynthesis' in window))
        return [];
    return window.speechSynthesis.getVoices();
}

export function setVoiceGender(g: Gender): void {
    wantGender = g;
}

// 当前所选性别在本机是否有对应的中文语音（语音列表可能异步加载，界面应在 voiceschanged 后再问一次）
export function voiceAvailable(g: Gender): boolean {
    return pickVoice(availableVoices(), g).matched;
}

export function onVoicesChanged(cb: () => void): () => void {
    if (!('speechSynthesis' in window))
        return () => undefined;
    window.speechSynthesis.addEventListener('voiceschanged', cb);
    return () => window.speechSynthesis.removeEventListener('voiceschanged', cb);
}

export function speak(text: string): void {
    if (!('speechSynthesis' in window))
        return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'zh-CN';
    u.rate = 1.05;
    const { voice, matched } = pickVoice(availableVoices(), wantGender);
    if (voice)
        u.voice = voice;
    // 本机没有所选性别的中文语音时，用音调粗略区分
    if (!matched)
        u.pitch = wantGender === 'male' ? 0.7 : 1.2;
    window.speechSynthesis.speak(u);
}

export function stopSpeaking(): void {
    if ('speechSynthesis' in window)
        window.speechSynthesis.cancel();
}

let audio: AudioContext | null = null;

// 必须在用户点击后首次调用，浏览器才允许发声
export function unlockAudio(): void {
    if (audio === null && 'AudioContext' in window)
        audio = new AudioContext();
    void audio?.resume();
}

export function beep(high = false): void {
    if (audio === null)
        return;
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.frequency.value = high ? 1320 : 880;
    gain.gain.setValueAtTime(0.2, audio.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.18);
    osc.connect(gain);
    gain.connect(audio.destination);
    osc.start();
    osc.stop(audio.currentTime + 0.2);
}
