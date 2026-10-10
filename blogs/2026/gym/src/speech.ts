// 语音播报与提示音。浏览器不支持时静默跳过，不影响计时

export function speak(text: string): void {
    if (!('speechSynthesis' in window))
        return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'zh-CN';
    u.rate = 1.05;
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
