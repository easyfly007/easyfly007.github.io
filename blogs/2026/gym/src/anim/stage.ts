import * as THREE from 'three';
import { EXERCISE_BY_ID } from '../data/exercises';
import type { Gender } from '../types';
import { createFigure, type Figure } from './figure';
import { ANIMATIONS } from './library';
import { mirrorDef, poseAt } from './pose';
import { ANIM_VIEW_LABEL, animCycleSec, type AnimView } from './schedule';
import type { AnimDef } from './types';

// 动作动画的 3D 舞台：场景、灯光、地面、三个机位。由跟练播放器和动画预览页共用

export interface Stage {
    setExercise: (exerciseId: string, side: 'left' | 'right' | undefined, coach: Gender) => void;
    // t：这个动作已进行的秒数；view：第几个视角（0..2）。返回当前视角名
    render: (t: number, view: number) => string;
    resize: () => void;
    dispose: () => void;
}

interface Framing {
    target: THREE.Vector3;
    halfV: number;              // 需要看到的半高
    halfH: number;              // 需要看到的半宽
    views: AnimView[];
}

const FOV = 32;

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
    const mat = new THREE.Mesh(
        new THREE.BoxGeometry(0.62, 0.008, 1.8),
        new THREE.MeshStandardMaterial({ color: 0x3d5a80, roughness: 0.8 }),
    );
    mat.receiveShadow = true;
    scene.add(mat);

    const camera = new THREE.PerspectiveCamera(FOV, 16 / 9, 0.1, 50);

    let figure: Figure | null = null;
    let def: AnimDef | null = null;
    let cycleSec = 2;
    let framing: Framing = { target: new THREE.Vector3(0, 0.9, 0), halfV: 1.1, halfH: 0.8, views: ['front', 'side', 'full'] };

    function clear() {
        if (!figure)
            return;
        scene.remove(figure.root);
        scene.remove(figure.props);
        figure = null;
    }

    // 采样整个循环，取身体范围，决定机位距离和中心；躺/撑的动作先看侧面
    function frame(f: Figure, d: AnimDef): Framing {
        const box = new THREE.Box3();
        let upright = true;
        for (let i = 0; i < 12; i++) {
            f.setPose(poseAt(d, i / 12), i / 12);
            const j = f.joints();
            for (const p of Object.values(j))
                box.expandByPoint(p);
            if (i === 0)
                upright = j.neck.y - j.pelvis.y > 0.3;
        }
        box.min.y = 0;
        const size = box.getSize(new THREE.Vector3());
        const target = box.getCenter(new THREE.Vector3());
        target.y = Math.max(0.45, size.y * 0.5);
        const halfV = size.y * 0.5 + 0.25;
        const halfH = Math.max(size.x, size.z) * 0.5 + 0.3;
        // 站姿先看正面；躺/撑时从能看到身体全长的方向先看（身体沿 z 方向躺就从侧面看，沿 x 方向侧躺就从正面看）
        let views: AnimView[] = ['front', 'side', 'full'];
        if (!upright)
            views = size.x > size.z ? ['front', 'full', 'side'] : ['side', 'full', 'front'];
        return { target, halfV, halfH, views };
    }

    function setExercise(exerciseId: string, side: 'left' | 'right' | undefined, coach: Gender) {
        clear();
        const base = ANIMATIONS[exerciseId];
        if (!base)
            return;
        def = side === 'right' ? mirrorDef(base) : base;
        cycleSec = animCycleSec(exerciseId);
        figure = createFigure(coach, def);
        scene.add(figure.root);
        scene.add(figure.props);
        framing = frame(figure, def);
        const usesMat = EXERCISE_BY_ID[exerciseId].equipment.includes('mat');
        mat.visible = usesMat;
        if (usesMat) {
            figure.setPose(poseAt(def, 0), 0);
            const j = figure.joints();
            const spanZ = Math.abs(j.head.z - j.ankleL.z) > Math.abs(j.head.x - j.ankleL.x);
            mat.rotation.y = spanZ ? 0 : Math.PI / 2;
            mat.position.set((j.head.x + j.ankleL.x) / 2, 0.004, (j.head.z + j.ankleL.z) / 2);
        }
    }

    function placeCamera(view: AnimView) {
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

    function render(t: number, view: number): string {
        const v = framing.views[view % 3];
        if (figure && def) {
            const phase = (t / cycleSec) % 1;
            figure.setPose(poseAt(def, phase), phase);
        }
        placeCamera(v);
        renderer.render(scene, camera);
        return ANIM_VIEW_LABEL[v];
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

    function dispose() {
        clear();
        renderer.dispose();
    }

    resize();
    return { setExercise, render, resize, dispose };
}
