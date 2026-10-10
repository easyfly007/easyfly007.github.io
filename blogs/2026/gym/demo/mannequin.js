// 简洁的低多边形人偶（胶囊体 + 球体拼成），分层级的关节 Group，程序化做深蹲。
// THREE 由调用方传入（浏览器页面和 node 校验脚本共用）。人物面朝 +z，y 向上，脚底在 y = 0

const DEG = Math.PI / 180;

// 男女体型：男教练肩宽、胸厚；女教练肩窄、髋略宽、身材略小，扎马尾
const BUILDS = {
    male: { scale: 1, shoulderX: 0.21, chestW: 1.2, hipX: 0.11, pelvisW: 1.1, shirt: 0xff6b35, pants: 0x34495e, ponytail: false },
    female: { scale: 0.94, shoulderX: 0.18, chestW: 1.05, hipX: 0.115, pelvisW: 1.2, shirt: 0x2bb3a0, pants: 0x3b3b58, ponytail: true },
};

export function createMannequin(THREE, gender) {
    const b = BUILDS[gender];
    const THIGH = 0.44;
    const SHIN = 0.44;
    const ANKLE_Y = 0.07;
    const TORSO = 0.52;
    const UPPER_ARM = 0.3;
    const FOREARM = 0.28;

    const skin = new THREE.MeshStandardMaterial({ color: 0xf2c6a0, roughness: 0.8, flatShading: true });
    const shirt = new THREE.MeshStandardMaterial({ color: b.shirt, roughness: 0.7, flatShading: true });
    const pants = new THREE.MeshStandardMaterial({ color: b.pants, roughness: 0.8, flatShading: true });
    const shoe = new THREE.MeshStandardMaterial({ color: 0x222831, roughness: 0.6, flatShading: true });
    const hair = new THREE.MeshStandardMaterial({ color: 0x2b2420, roughness: 0.9, flatShading: true });

    function mesh(geo, mat) {
        const m = new THREE.Mesh(geo, mat);
        m.castShadow = true;
        return m;
    }

    // 沿 -y 方向的肢体段（关节在原点）
    function limb(length, radius, mat) {
        const m = mesh(new THREE.CapsuleGeometry(radius, length - 2 * radius, 4, 8), mat);
        m.position.y = -length / 2;
        return m;
    }

    const root = new THREE.Group();
    root.scale.setScalar(b.scale);
    const pelvis = new THREE.Group();
    root.add(pelvis);
    pelvis.add(mesh(new THREE.SphereGeometry(0.15, 10, 8).scale(b.pelvisW, 0.7, 0.8), pants));

    const torso = new THREE.Group();
    pelvis.add(torso);
    const chest = mesh(new THREE.CapsuleGeometry(0.14, TORSO - 0.2, 4, 10).scale(b.chestW, 1, 0.75), shirt);
    chest.position.y = TORSO / 2;
    torso.add(chest);
    const neck = mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.08, 8), skin);
    neck.position.y = TORSO + 0.03;
    torso.add(neck);
    const head = mesh(new THREE.SphereGeometry(0.11, 12, 10).scale(0.9, 1.05, 0.95), skin);
    head.position.y = TORSO + 0.17;
    torso.add(head);
    const cap = mesh(new THREE.SphereGeometry(0.112, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.45).scale(0.92, 1.05, 0.98), hair);
    cap.position.y = TORSO + 0.175;
    torso.add(cap);
    if (b.ponytail) {
        const tail = mesh(new THREE.CapsuleGeometry(0.035, 0.12, 4, 8), hair);
        tail.position.set(0, TORSO + 0.13, -0.12);
        tail.rotation.x = -0.35;
        torso.add(tail);
    }

    function makeArm(side) {
        const shoulder = new THREE.Group();
        shoulder.position.set(side * b.shoulderX, TORSO - 0.06, 0);
        torso.add(shoulder);
        shoulder.add(mesh(new THREE.SphereGeometry(0.055, 8, 6), shirt));
        shoulder.add(limb(UPPER_ARM, 0.045, shirt));
        const elbow = new THREE.Group();
        elbow.position.y = -UPPER_ARM;
        shoulder.add(elbow);
        elbow.add(limb(FOREARM, 0.04, skin));
        const hand = mesh(new THREE.SphereGeometry(0.045, 8, 6), skin);
        hand.position.y = -FOREARM - 0.02;
        elbow.add(hand);
        return { shoulder, elbow };
    }

    function makeLeg(side) {
        const hip = new THREE.Group();
        hip.position.set(side * b.hipX, 0, 0);
        pelvis.add(hip);
        hip.add(limb(THIGH, 0.07, pants));
        const knee = new THREE.Group();
        knee.position.y = -THIGH;
        hip.add(knee);
        knee.add(limb(SHIN, 0.055, pants));
        const ankle = new THREE.Group();
        ankle.position.y = -SHIN;
        knee.add(ankle);
        const foot = mesh(new THREE.BoxGeometry(0.1, 0.07, 0.24), shoe);
        foot.position.set(0, -ANKLE_Y / 2, 0.06);
        ankle.add(foot);
        return { hip, knee, ankle };
    }

    const arms = [makeArm(-1), makeArm(1)];
    const legs = [makeLeg(-1), makeLeg(1)];

    // 深蹲姿态：脚固定，两段腿几何求髋位置。s ∈ [0, 1]：0 站直，1 蹲到底
    function pose(s) {
        const shinAngle = 32 * DEG * s;     // 小腿前倾（膝盖向前，朝脚尖方向）
        const thighAngle = 88 * DEG * s;    // 大腿后倒（髋向后坐）
        const lean = 38 * DEG * s;          // 躯干前倾
        // 侧视平面 (z 向前, y 向上)，踝关节固定在 (0, ANKLE_Y)
        const kneeZ = SHIN * Math.sin(shinAngle);
        const kneeY = ANKLE_Y + SHIN * Math.cos(shinAngle);
        const hipZ = kneeZ - THIGH * Math.sin(thighAngle);
        const hipY = kneeY + THIGH * Math.cos(thighAngle);
        pelvis.position.set(0, hipY, hipZ);
        torso.rotation.x = lean;
        for (const leg of legs) {
            // 大腿方向 (sin b, -cos b) 对应绕 x 旋转 -b；小腿世界角为 a，膝关节相对角 a + b；脚掌保持水平
            leg.hip.rotation.x = -thighAngle;
            leg.knee.rotation.x = thighAngle + shinAngle;
            leg.ankle.rotation.x = -shinAngle;
        }
        // 手臂前平举配重：世界角从下垂 0 到前平举 90°，扣掉躯干前倾
        const raise = 90 * DEG * s;
        for (const arm of arms) {
            arm.shoulder.rotation.x = -raise - lean;
            arm.elbow.rotation.x = -10 * DEG * s;
        }
    }

    pose(0);
    const height = (ANKLE_Y + SHIN + THIGH + TORSO + 0.28) * b.scale;
    return { root, pose, height };
}

// 每次动作的深度曲线：首尾都是站直（0），中间蹲到底（1），所以每次首尾姿态相同
export function squatDepth(phase) {
    return (1 - Math.cos(2 * Math.PI * phase)) / 2;
}
