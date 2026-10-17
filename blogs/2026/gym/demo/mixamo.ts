import * as THREE from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';

// Mixamo 素材预览：直接加载 assets/mixamo/ 下的人物 FBX（带皮）和动作 FBX（不带皮），
// 把动作套到人物上播放，用跟练播放器同样的三个机位看效果，并帮用户找出"一次动作"的起止帧。
// 这是验证用的页面，不进正式代码；正式管线见 docs/mixamo.md

export const FPS = 30;
export type View = 'front' | 'side' | 'full';
export const VIEW_LABEL: Record<View, string> = { front: '正面', side: '侧面', full: '全身' };

// 放在 assets/mixamo/ 下的文件由 Vite 自动收录，不需要登记
const CHARACTER_FILES = import.meta.glob('/assets/mixamo/characters/*.fbx', { eager: true, query: '?url', import: 'default' }) as Record<string, string>;
const ANIM_FILES = import.meta.glob('/assets/mixamo/anims/*.fbx', { eager: true, query: '?url', import: 'default' }) as Record<string, string>;

function baseName(path: string): string {
    return path.split('/').pop()!.replace(/\.fbx$/i, '');
}

export function listCharacters(): { name: string; url: string }[] {
    return Object.entries(CHARACTER_FILES).map(([p, url]) => ({ name: baseName(p), url })).sort((a, b) => a.name.localeCompare(b.name));
}

export function listAnims(): { name: string; url: string }[] {
    return Object.entries(ANIM_FILES).map(([p, url]) => ({ name: baseName(p), url })).sort((a, b) => a.name.localeCompare(b.name));
}

const loader = new FBXLoader();

export interface Character {
    root: THREE.Group;
    mixer: THREE.AnimationMixer;
    boneNames: Set<string>;
    height: number;             // 米
}

function loadFbx(url: string, onProgress?: (ratio: number) => void): Promise<THREE.Group> {
    return new Promise((resolve, reject) => {
        loader.load(url, resolve, (e) => {
            if (onProgress && e.total > 0)
                onProgress(e.loaded / e.total);
        }, reject);
    });
}

export async function loadCharacter(url: string, onProgress?: (ratio: number) => void): Promise<Character> {
    const root = await loadFbx(url, onProgress);
    // Mixamo 的单位是厘米
    root.scale.setScalar(0.01);
    const boneNames = new Set<string>();
    root.traverse((o) => {
        if ((o as THREE.Bone).isBone)
            boneNames.add(o.name);
        const mesh = o as THREE.Mesh;
        if (mesh.isMesh) {
            mesh.castShadow = true;
            mesh.frustumCulled = false;
            const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
            for (const m of mats)
                m.side = THREE.DoubleSide;
        }
    });
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(root);
    return { root, mixer: new THREE.AnimationMixer(root), boneNames, height: box.max.y - box.min.y };
}

// 动作文件里的骨骼名和人物不一致时，把轨道名对到人物的骨骼上。
// Mixamo 的前缀不固定：动作文件是 mixamorig，人物可能是 mixamorig5、mixamorig: 等
function retarget(clip: THREE.AnimationClip, boneNames: Set<string>): THREE.AnimationClip {
    const strip = (n: string) => n.replace(/^mixamorig\d*:?/, '');
    const byStripped = new Map<string, string>();
    for (const n of boneNames)
        byStripped.set(strip(n), n);
    const tracks: THREE.KeyframeTrack[] = [];
    for (const t of clip.tracks) {
        const dot = t.name.lastIndexOf('.');
        const node = t.name.slice(0, dot);
        const prop = t.name.slice(dot);
        if (boneNames.has(node)) {
            tracks.push(t);
            continue;
        }
        const mapped = byStripped.get(strip(node));
        if (!mapped)
            continue;
        const copy = t.clone();
        copy.name = mapped + prop;
        tracks.push(copy);
    }
    return new THREE.AnimationClip(clip.name, clip.duration, tracks);
}

export async function loadAnim(url: string, ch: Character): Promise<THREE.AnimationClip> {
    const group = await loadFbx(url);
    const clip = group.animations[0];
    if (!clip)
        throw new Error('这个文件里没有动画');
    return retarget(clip, ch.boneNames);
}

export function frameCount(clip: THREE.AnimationClip): number {
    return Math.round(clip.duration * FPS);
}

// 把人物摆到动画的某一帧
export function setFrame(ch: Character, action: THREE.AnimationAction, frame: number) {
    action.time = Math.max(0, frame) / FPS;
    ch.mixer.update(0);
    ch.root.updateMatrixWorld(true);
}

export interface Framing {
    target: THREE.Vector3;
    halfV: number;
    halfH: number;
    views: View[];
}

// 采样整段动画，取身体范围，决定机位距离和中心；躺/撑的动作先看侧面（与 src/anim/stage.ts 同样的规则）
export function frameAnimation(ch: Character, action: THREE.AnimationAction, start: number, end: number): Framing {
    const box = new THREE.Box3();
    const hips = ch.root.getObjectByName([...ch.boneNames].find((n) => /Hips$/.test(n)) ?? '');
    const head = ch.root.getObjectByName([...ch.boneNames].find((n) => /Head$/.test(n)) ?? '');
    let upright = true;
    const p = new THREE.Vector3();
    for (let i = 0; i < 12; i++) {
        setFrame(ch, action, start + ((end - start) * i) / 12);
        ch.root.traverse((o) => {
            if ((o as THREE.Bone).isBone)
                box.expandByPoint(o.getWorldPosition(p));
        });
        if (i === 0 && hips && head)
            upright = head.getWorldPosition(new THREE.Vector3()).y - hips.getWorldPosition(new THREE.Vector3()).y > 0.3;
    }
    box.min.y = Math.min(box.min.y, 0);
    const size = box.getSize(new THREE.Vector3());
    const target = box.getCenter(new THREE.Vector3());
    target.y = Math.max(0.45, size.y * 0.5);
    let views: View[] = ['front', 'side', 'full'];
    if (!upright)
        views = size.x > size.z ? ['front', 'full', 'side'] : ['side', 'full', 'front'];
    return { target, halfV: size.y * 0.5 + 0.25, halfH: Math.max(size.x, size.z) * 0.5 + 0.3, views };
}

const FOV = 32;

export interface Stage {
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    place: (framing: Framing, view: View) => void;
    render: () => void;
    resize: () => void;
}

export function createStage(canvas: HTMLCanvasElement, darkMode: boolean): Stage {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(darkMode ? 0x1e2127 : 0xeef1f5);
    scene.add(new THREE.HemisphereLight(0xffffff, darkMode ? 0x30343c : 0xb9c0cc, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 4, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = -2.5;
    sun.shadow.camera.right = 2.5;
    sun.shadow.camera.top = 2.5;
    sun.shadow.camera.bottom = -2.5;
    scene.add(sun);

    const floor = new THREE.Mesh(
        new THREE.CircleGeometry(3.2, 48),
        new THREE.MeshStandardMaterial({ color: darkMode ? 0x2a2e35 : 0xffffff, roughness: 1 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    const grid = new THREE.GridHelper(6.4, 32, darkMode ? 0x4a505a : 0xc8cdd5, darkMode ? 0x353a42 : 0xdde1e7);
    grid.position.y = 0.001;
    scene.add(grid);

    const camera = new THREE.PerspectiveCamera(FOV, 16 / 9, 0.1, 50);

    function place(framing: Framing, view: View) {
        const { target: t, halfV, halfH } = framing;
        const d = Math.max(halfV, halfH / camera.aspect) / Math.tan((FOV / 2) * Math.PI / 180);
        if (view === 'front')
            camera.position.set(t.x, t.y + d * 0.08, t.z + d);
        else if (view === 'side')
            camera.position.set(t.x + d, t.y + d * 0.08, t.z);
        else
            camera.position.set(t.x + d * 0.62, t.y + d * 0.55, t.z + d * 0.72);
        camera.lookAt(t);
    }

    function resize() {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (w === 0 || h === 0)
            return;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }

    resize();
    return { scene, camera, place, render: () => renderer.render(scene, camera), resize };
}
