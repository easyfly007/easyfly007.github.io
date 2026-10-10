import { describe, expect, it } from 'vitest';
import { pickVoice, voiceGender } from './speech';

function voice(name: string, lang = 'zh-CN'): SpeechSynthesisVoice {
    return { name, lang, default: false, localService: true, voiceURI: name };
}

describe('voiceGender', () => {
    it('recognises common Chinese system voices', () => {
        expect(voiceGender(voice('Microsoft Xiaoxiao Online (Natural) - Chinese (Mainland)'))).toBe('female');
        expect(voiceGender(voice('Microsoft Yunxi Online (Natural) - Chinese (Mainland)'))).toBe('male');
        expect(voiceGender(voice('Microsoft Huihui - Chinese (Simplified, PRC)'))).toBe('female');
        expect(voiceGender(voice('Microsoft Kangkang - Chinese (Simplified, PRC)'))).toBe('male');
        expect(voiceGender(voice('Tingting'))).toBe('female');
        expect(voiceGender(voice('Li-mu'))).toBe('male');
        expect(voiceGender(voice('Google 普通话（中国大陆）'))).toBe('female');
        expect(voiceGender(voice('Some Female Voice'))).toBe('female');
        expect(voiceGender(voice('Unknown'))).toBeNull();
    });
});

describe('pickVoice', () => {
    const voices = [
        voice('Samantha', 'en-US'),
        voice('Sin-ji', 'zh-HK'),
        voice('Microsoft Kangkang', 'zh-CN'),
        voice('Microsoft Huihui', 'zh-CN'),
    ];

    it('picks a Mandarin voice of the requested gender', () => {
        expect(pickVoice(voices, 'female')).toEqual({ voice: voices[3], matched: true });
        expect(pickVoice(voices, 'male')).toEqual({ voice: voices[2], matched: true });
    });

    it('falls back to any Chinese voice when the gender is missing', () => {
        const onlyFemale = [voices[0], voices[3]];
        expect(pickVoice(onlyFemale, 'male')).toEqual({ voice: voices[3], matched: false });
        expect(pickVoice([voices[0]], 'male')).toEqual({ voice: null, matched: false });
    });
});
