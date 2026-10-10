// mulberry32：小而快的可复现伪随机数，同一个种子永远得到同一串数
export function createRng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function newSeed(): number {
    return Math.floor(Math.random() * 0xffffffff);
}
