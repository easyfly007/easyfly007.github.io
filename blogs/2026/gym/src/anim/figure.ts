import * as THREE from 'three';
import type { Gender } from '../types';
import { fullPose } from './pose';
import type { AnimDef, BodyProp, Pose, SceneProp, V3 } from './types';

// 简洁的低多边形人偶（胶囊体 + 球体），每段肢体是一个 Group，按姿态里的方向摆放。
// 摆好后自动贴地（最低点落在地面），并让 anchor 指定的部位在水平面上保持不动

const DOWN = new THREE.Vector3(0, -1, 0);
const UP = new THREE.Vector3(0, 1, 0);
const RAD = Math.PI / 180;

const THIGH = 0.44;
const SHIN = 0.44;
const ANKLE = 0.07;
const TORSO = 0.52;
const UPPER = 0.3;
const FORE = 0.28;
const BAR_HEIGHT = 2.15;

// 男教练肩宽、胸厚；女教练肩窄、髋略宽、身材略小，扎马尾
const BUILDS = {
    male: { scale: 1, shoulderX: 0.21, chestW: 1.2, hipX: 0.11, pelvisW: 1.1, shirt: 0xff6b35, pants: 0x34495e, ponytail: false },
    female: { scale: 0.94, shoulderX: 0.18, chestW: 1.05, hipX: 0.115, pelvisW: 1.2, shirt: 0x2bb3a0, pants: 0x3b3b58, ponytail: true },
};

function v3(v: V3): THREE.Vector3 {
    return new THREE.Vector3(v[0], v[1], v[2]).normalize();
}

interface Limb {
    upper: THREE.Group;
    lower: THREE.Group;
    end: THREE.Object3D;        // 手 / 脚踝
}

export interface Figure {
    root: THREE.Group;          // 人偶（加到场景里）
    props: THREE.Group;         // 器械与场景物体（加到场景里）
    height: number;
    setPose: (p: Pose, phase?: number) => void;
    joints: () => Record<string, THREE.Vector3>;
}

function material(color: number): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({ color, roughness: 0.75, flatShading: true });
}

export function createFigure(gender: Gender, def: AnimDef): Figure {
    const b = BUILDS[gender];
    const skin = material(0xf2c6a0);
    const shirt = material(b.shirt);
    const pants = material(b.pants);
    const shoe = material(0x222831);
    const hair = material(0x2b2420);
    const bodyMeshes: THREE.Mesh[] = [];

    function mesh(geo: THREE.BufferGeometry, mat: THREE.Material, parent: THREE.Object3D): THREE.Mesh {
        const m = new THREE.Mesh(geo, mat);
        m.castShadow = true;
        parent.add(m);
        bodyMeshes.push(m);
        return m;
    }

    // 沿 -y 方向的一段肢体，关节在原点
    function segment(parent: THREE.Object3D, at: THREE.Vector3, length: number, radius: number, mat: THREE.Material): THREE.Group {
        const g = new THREE.Group();
        g.position.copy(at);
        parent.add(g);
        const m = mesh(new THREE.CapsuleGeometry(radius, length - 2 * radius, 4, 8), mat, g);
        m.position.y = -length / 2;
        return g;
    }

    const root = new THREE.Group();
    const body = new THREE.Group();     // 骨盆：姿态的 pitch / roll / yaw 作用在这里
    body.scale.setScalar(b.scale);
    root.add(body);
    mesh(new THREE.SphereGeometry(0.15, 10, 8).scale(b.pelvisW, 0.7, 0.8), pants, body);

    const chest = new THREE.Group();
    body.add(chest);
    const torsoMesh = mesh(new THREE.CapsuleGeometry(0.14, TORSO - 0.2, 4, 10).scale(b.chestW, 1, 0.75), shirt, chest);
    torsoMesh.position.y = TORSO / 2;

    const head = new THREE.Group();
    head.position.y = TORSO;
    chest.add(head);
    mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.08, 8), skin, head).position.y = 0.03;
    mesh(new THREE.SphereGeometry(0.11, 12, 10).scale(0.9, 1.05, 0.95), skin, head).position.y = 0.17;
    mesh(new THREE.SphereGeometry(0.112, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.45).scale(0.92, 1.05, 0.98), hair, head).position.y = 0.175;
    if (b.ponytail) {
        const tail = mesh(new THREE.CapsuleGeometry(0.035, 0.12, 4, 8), hair, head);
        tail.position.set(0, 0.13, -0.12);
        tail.rotation.x = -0.35;
    }

    function makeArm(side: number): Limb {
        const at = new THREE.Vector3(side * b.shoulderX, TORSO - 0.06, 0);
        const upper = segment(chest, at, UPPER, 0.045, shirt);
        mesh(new THREE.SphereGeometry(0.055, 8, 6), shirt, upper);
        const lower = segment(upper, new THREE.Vector3(0, -UPPER, 0), FORE, 0.04, skin);
        const hand = new THREE.Object3D();
        hand.position.y = -FORE - 0.02;
        lower.add(hand);
        mesh(new THREE.SphereGeometry(0.045, 8, 6), skin, hand);
        return { upper, lower, end: hand };
    }

    function makeLeg(side: number): Limb & { foot: THREE.Group } {
        const upper = segment(body, new THREE.Vector3(side * b.hipX, 0, 0), THIGH, 0.07, pants);
        const lower = segment(upper, new THREE.Vector3(0, -THIGH, 0), SHIN, 0.055, pants);
        const foot = new THREE.Group();
        foot.position.y = -SHIN;
        lower.add(foot);
        const box = mesh(new THREE.BoxGeometry(0.1, ANKLE, 0.24), shoe, foot);
        box.position.set(0, -ANKLE / 2, 0.06);
        return { upper, lower, end: foot, foot };
    }

    const arms = { L: makeArm(1), R: makeArm(-1) };
    const legsL = makeLeg(1);
    const legsR = makeLeg(-1);
    const legs = { L: legsL, R: legsR };

    const props = new THREE.Group();
    const propParts = buildBodyProps(def.props ?? [], props);
    let sceneProps: THREE.Object3D[] | null = null;

    const q = new THREE.Quaternion();
    const qa = new THREE.Quaternion();
    const tmp = new THREE.Vector3();

    // 让 g 的 -y 轴指向 dir（dir 在 frame 的坐标系里，frame 为 null 表示世界坐标）
    function point(g: THREE.Object3D, dir: V3, frame: THREE.Object3D | null) {
        const d = v3(dir);
        if (frame)
            d.applyQuaternion(frame.getWorldQuaternion(qa));
        g.parent!.getWorldQuaternion(q);
        d.applyQuaternion(q.invert());
        g.quaternion.setFromUnitVectors(DOWN, d);
        g.updateMatrixWorld(true);
    }

    function bodyIsUpright(): boolean {
        return UP.clone().applyQuaternion(body.getWorldQuaternion(qa)).y > 0.6;
    }

    // 脚：脚尖朝 toe（世界坐标）；不给时站姿脚掌放平朝前，躺/撑时脚尖顺着身体
    function placeFoot(foot: THREE.Group, toe: V3 | undefined) {
        const shin = foot.parent!;
        const shinUp = UP.clone().applyQuaternion(shin.getWorldQuaternion(qa));
        let z: THREE.Vector3;
        if (toe)
            z = v3(toe);
        else {
            const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(body.getWorldQuaternion(qa));
            if (bodyIsUpright())
                z = new THREE.Vector3(fwd.x, 0, fwd.z).normalize();
            else
                z = fwd.sub(shinUp.clone().multiplyScalar(fwd.dot(shinUp))).normalize();
        }
        const y = shinUp.sub(z.clone().multiplyScalar(shinUp.dot(z)));
        if (y.lengthSq() < 1e-6)
            y.set(0, 1, 0);
        y.normalize();
        const x = new THREE.Vector3().crossVectors(y, z);
        const m = new THREE.Matrix4().makeBasis(x, y, z);
        const world = new THREE.Quaternion().setFromRotationMatrix(m);
        shin.getWorldQuaternion(q);
        foot.quaternion.copy(q.invert().multiply(world));
        foot.updateMatrixWorld(true);
    }

    const world = (o: THREE.Object3D) => o.getWorldPosition(new THREE.Vector3());

    function anchorPoint(): THREE.Vector3 | null {
        switch (def.anchor) {
            case 'feet':
                return world(legs.L.foot).add(world(legs.R.foot)).multiplyScalar(0.5);
            case 'footL':
                return world(legs.L.foot);
            case 'footR':
                return world(legs.R.foot);
            case 'hands':
                return world(arms.L.end).add(world(arms.R.end)).multiplyScalar(0.5);
            default:
                return null;
        }
    }

    const box = new THREE.Box3();
    function bodyMinY(): number {
        box.makeEmpty();
        for (const m of bodyMeshes)
            box.expandByObject(m, true);
        return box.min.y;
    }

    function setPose(input: Pose, phase = 0) {
        const p = fullPose(input);
        root.position.set(0, 0, 0);
        body.position.set(0, 0, 0);
        body.quaternion.setFromEuler(new THREE.Euler((p.pitch ?? 0) * RAD, (p.yaw ?? 0) * RAD, -(p.roll ?? 0) * RAD, 'YXZ'));
        body.updateMatrixWorld(true);

        chest.quaternion.setFromUnitVectors(UP, v3(p.spine!));
        chest.quaternion.multiply(qa.setFromAxisAngle(UP, (p.twist ?? 0) * RAD));
        chest.updateMatrixWorld(true);
        head.quaternion.setFromUnitVectors(UP, v3(p.head!));
        head.updateMatrixWorld(true);

        const armFrame = p.armsWorld ? null : chest;
        const legFrame = p.legsWorld ? null : body;
        for (const s of ['L', 'R'] as const) {
            point(arms[s].upper, p[`arm${s}`]!, armFrame);
            point(arms[s].lower, p[`fore${s}`]!, armFrame);
            point(legs[s].upper, p[`thigh${s}`]!, legFrame);
            point(legs[s].lower, p[`shin${s}`]!, legFrame);
            placeFoot(legs[s].foot, p[`toe${s}`]);
        }
        root.updateMatrixWorld(true);

        if (def.hang) {
            // 悬垂：两手中点固定在单杠高度
            const hands = world(arms.L.end).add(world(arms.R.end)).multiplyScalar(0.5);
            root.position.set(-hands.x, BAR_HEIGHT - hands.y, -hands.z);
        } else {
            const a = anchorPoint();
            root.position.y = -bodyMinY() + (p.lift ?? 0);
            if (a) {
                root.position.x = -a.x;
                root.position.z = -a.z;
            }
        }
        root.updateMatrixWorld(true);

        if (!sceneProps)
            sceneProps = buildSceneProps(def.scene ?? [], props, joints());
        updateBodyProps(propParts, joints(), chest, phase);
    }

    function joints(): Record<string, THREE.Vector3> {
        return {
            pelvis: world(body),
            neck: world(head),
            head: head.localToWorld(new THREE.Vector3(0, 0.17, 0)),
            shoulderL: world(arms.L.upper),
            shoulderR: world(arms.R.upper),
            elbowL: world(arms.L.lower),
            elbowR: world(arms.R.lower),
            handL: world(arms.L.end),
            handR: world(arms.R.end),
            hipL: world(legs.L.upper),
            hipR: world(legs.R.upper),
            kneeL: world(legs.L.lower),
            kneeR: world(legs.R.lower),
            ankleL: world(legs.L.foot),
            ankleR: world(legs.R.foot),
            toeL: legs.L.foot.localToWorld(tmp.set(0, -ANKLE / 2, 0.18).clone()),
            toeR: legs.R.foot.localToWorld(tmp.set(0, -ANKLE / 2, 0.18).clone()),
            chestBack: chest.localToWorld(new THREE.Vector3(0, TORSO * 0.55, -0.12)),
        };
    }

    setPose(def.keys[0][1]);
    const height = (ANKLE + SHIN + THIGH + TORSO + 0.28) * b.scale;
    return { root, props, height, setPose, joints };
}

// ---------- 器械 ----------

const metal = () => new THREE.MeshStandardMaterial({ color: 0x5b6470, roughness: 0.4, metalness: 0.6 });
const dark = () => new THREE.MeshStandardMaterial({ color: 0x2c2f36, roughness: 0.6 });

function dumbbell(): THREE.Group {
    const g = new THREE.Group();
    const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.16, 8), metal());
    handle.rotation.z = Math.PI / 2;
    g.add(handle);
    for (const s of [-1, 1]) {
        const w = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.06, 10), dark());
        w.rotation.z = Math.PI / 2;
        w.position.x = s * 0.1;
        g.add(w);
    }
    g.traverse((o) => {
        o.castShadow = true;
    });
    return g;
}

function kettlebell(): THREE.Group {
    const g = new THREE.Group();
    const bell = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 10), dark());
    bell.position.y = -0.11;
    g.add(bell);
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.05, 0.012, 6, 12, Math.PI), metal());
    handle.position.y = -0.04;
    g.add(handle);
    g.traverse((o) => {
        o.castShadow = true;
    });
    return g;
}

function barbell(): THREE.Group {
    const g = new THREE.Group();
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 1.8, 8), metal());
    bar.rotation.z = Math.PI / 2;
    g.add(bar);
    for (const s of [-1, 1]) {
        const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.05, 16), dark());
        plate.rotation.z = Math.PI / 2;
        plate.position.x = s * 0.72;
        g.add(plate);
    }
    g.traverse((o) => {
        o.castShadow = true;
    });
    return g;
}

const bandMaterial = () => new THREE.MeshStandardMaterial({ color: 0xe0531f, roughness: 0.6 });

// 两点之间的一根弹力带 / 绳子
function strap(parent: THREE.Group, color?: THREE.Material): (a: THREE.Vector3, b: THREE.Vector3) => void {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 6), color ?? bandMaterial());
    parent.add(m);
    return (a, b) => {
        const d = b.clone().sub(a);
        m.position.copy(a).add(b).multiplyScalar(0.5);
        m.scale.set(1, Math.max(d.length(), 0.001), 1);
        m.quaternion.setFromUnitVectors(UP, d.normalize());
    };
}

type Updater = (j: Record<string, THREE.Vector3>, chest: THREE.Object3D, phase: number) => void;

function buildBodyProps(list: BodyProp[], parent: THREE.Group): Updater[] {
    const out: Updater[] = [];
    const sideAxis = (chest: THREE.Object3D) => new THREE.Vector3(1, 0, 0).applyQuaternion(chest.getWorldQuaternion(new THREE.Quaternion()));
    for (const kind of list) {
        if (kind === 'dumbbells' || kind === 'dumbbellL' || kind === 'dumbbellR') {
            const hands = kind === 'dumbbells' ? ['handL', 'handR'] : [kind === 'dumbbellL' ? 'handL' : 'handR'];
            const ds = hands.map(() => dumbbell());
            ds.forEach((d) => parent.add(d));
            out.push((j, chest) => {
                const axis = sideAxis(chest);
                ds.forEach((d, i) => {
                    d.position.copy(j[hands[i]]);
                    d.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), axis);
                });
            });
        } else if (kind === 'dumbbellGoblet') {
            const d = dumbbell();
            parent.add(d);
            out.push((j) => {
                d.position.copy(j.handL).add(j.handR).multiplyScalar(0.5);
                d.position.y -= 0.06;
                d.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), UP);
            });
        } else if (kind === 'kettlebellGoblet') {
            const k = kettlebell();
            parent.add(k);
            out.push((j) => {
                k.position.copy(j.handL).add(j.handR).multiplyScalar(0.5);
                k.position.y += 0.04;
                k.quaternion.identity();
            });
        } else if (kind === 'kettlebell') {
            const k = kettlebell();
            parent.add(k);
            out.push((j) => {
                k.position.copy(j.handL).add(j.handR).multiplyScalar(0.5);
                const down = j.handL.clone().add(j.handR).multiplyScalar(0.5).sub(j.shoulderL.clone().add(j.shoulderR).multiplyScalar(0.5)).normalize();
                k.quaternion.setFromUnitVectors(DOWN, down);
            });
        } else if (kind === 'barbell' || kind === 'barbellBack') {
            const bar = barbell();
            parent.add(bar);
            out.push((j, chest) => {
                if (kind === 'barbellBack')
                    bar.position.copy(j.chestBack);
                else
                    bar.position.copy(j.handL).add(j.handR).multiplyScalar(0.5);
                bar.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), sideAxis(chest));
            });
        } else if (kind === 'bandHands') {
            const s = strap(parent);
            out.push((j) => s(j.handL, j.handR));
        } else if (kind === 'bandKnees') {
            const s = strap(parent);
            out.push((j) => s(j.kneeL, j.kneeR));
        } else if (kind === 'bandFeet') {
            const l = strap(parent);
            const r = strap(parent);
            out.push((j) => {
                l(j.handL, j.ankleL);
                r(j.handR, j.ankleR);
            });
        } else if (kind === 'bandAnchor') {
            const l = strap(parent);
            const r = strap(parent);
            let anchor: THREE.Vector3 | null = null;
            out.push((j, chest) => {
                if (!anchor) {
                    const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(chest.getWorldQuaternion(new THREE.Quaternion()));
                    fwd.y = 0;
                    anchor = j.handL.clone().add(j.handR).multiplyScalar(0.5).add(fwd.normalize().multiplyScalar(1.2));
                }
                l(j.handL, anchor);
                r(j.handR, anchor);
            });
        } else if (kind === 'rope') {
            const rope = new THREE.Group();
            parent.add(rope);
            const pieces = Array.from({ length: 8 }, () => strap(rope, new THREE.MeshStandardMaterial({ color: 0x3b6fd8 })));
            out.push((j, _chest, phase) => {
                // 绳子每个循环绕身体转一圈：循环开始时在头顶，转到一半时在脚下（此时人腾空）
                const angle = phase * Math.PI * 2;
                const mid = j.handL.clone().add(j.handR).multiplyScalar(0.5);
                const radius = Math.max(0.9, j.head.y - j.ankleL.y + 0.15) / 2;
                const center = mid.clone().setY((j.head.y + j.ankleL.y) / 2);
                const pts: THREE.Vector3[] = [];
                for (let i = 0; i <= 8; i++) {
                    const t = i / 8;
                    const across = j.handL.clone().lerp(j.handR, t);
                    const bulge = Math.sin(Math.PI * t);
                    const off = new THREE.Vector3(0, Math.cos(angle) * radius, Math.sin(angle) * radius * 0.6).multiplyScalar(bulge);
                    pts.push(across.lerp(center, bulge * 0.9).add(off));
                }
                pieces.forEach((s, i) => s(pts[i], pts[i + 1]));
            });
        }
    }
    return out;
}

function updateBodyProps(list: Updater[], j: Record<string, THREE.Vector3>, chest: THREE.Object3D, phase: number) {
    for (const u of list)
        u(j, chest, phase);
}

function buildSceneProps(list: SceneProp[], parent: THREE.Group, j: Record<string, THREE.Vector3>): THREE.Object3D[] {
    const out: THREE.Object3D[] = [];
    const wood = new THREE.MeshStandardMaterial({ color: 0xb08a5a, roughness: 0.8 });
    const wallMat = new THREE.MeshStandardMaterial({ color: 0xcfd4da, roughness: 1, side: THREE.DoubleSide });
    for (const s of list) {
        const at = {
            hands: j.handL.clone().add(j.handR).multiplyScalar(0.5),
            handL: j.handL.clone(),
            footL: j.ankleL.clone(),
            footR: j.ankleR.clone(),
            back: j.chestBack.clone(),
            front: j.pelvis.clone(),
        }[s.at];
        if (s.offset)
            at.add(new THREE.Vector3(...s.offset));
        let o: THREE.Object3D;
        if (s.kind === 'chair') {
            // 凳面顶部对准指定位置
            const h = Math.max(0.1, at.y - 0.03);
            o = new THREE.Mesh(new THREE.BoxGeometry(0.45, h, 0.42), wood);
            o.position.set(at.x, h / 2, at.z);
        } else if (s.kind === 'bar') {
            o = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1.2, 8), metal());
            o.rotation.z = Math.PI / 2;
            o.position.copy(at);
        } else {
            o = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 2.4), wallMat);
            o.position.set(at.x, 1.2, at.z);
        }
        o.castShadow = true;
        o.receiveShadow = true;
        parent.add(o);
        out.push(o);
    }
    return out;
}
