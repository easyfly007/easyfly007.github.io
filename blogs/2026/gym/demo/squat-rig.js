// 用骨骼驱动 Mixamo 骨架的人物做深蹲。THREE 由调用方传入（浏览器和 node 校验脚本共用这份代码）。
// 做法与坐标轴无关：每帧先恢复 T 姿态，再把每段骨骼"对准"世界坐标里的目标方向，
// 脚固定在地面上，用两段腿的几何算出髋的位置，所以脚不会滑。
// 人物面朝 +z，y 向上。

const DEG = Math.PI / 180;

// 有的模型面朝 -z（比如 Soldier），先整体转过来，让人物面朝 +z
export function faceForward(THREE, root) {
    root.updateMatrixWorld(true);
    const foot = root.getObjectByName('mixamorigLeftFoot').getWorldPosition(new THREE.Vector3());
    const toe = root.getObjectByName('mixamorigLeftToeBase').getWorldPosition(new THREE.Vector3());
    if (toe.z < foot.z)
        root.rotation.y += Math.PI;
    root.updateMatrixWorld(true);
}

export function createSquatRig(THREE, root) {
    const bone = (name) => {
        const b = root.getObjectByName(`mixamorig${name}`);
        if (!b)
            throw new Error(`找不到骨骼 ${name}`);
        return b;
    };
    const B = {};
    for (const n of ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End']) {
        B[n] = bone(n);
    }
    for (const s of ['Left', 'Right']) {
        for (const n of ['UpLeg', 'Leg', 'Foot', 'ToeBase', 'Arm', 'ForeArm', 'Hand', 'HandMiddle1']) {
            B[s + n] = bone(s + n);
        }
    }

    const v1 = new THREE.Vector3();
    const v2 = new THREE.Vector3();
    const q1 = new THREE.Quaternion();
    const q2 = new THREE.Quaternion();
    const q3 = new THREE.Quaternion();
    const world = (b) => b.getWorldPosition(new THREE.Vector3());

    // 记录静止（T 姿态）时每根骨骼的局部变换和若干世界量
    root.updateMatrixWorld(true);
    const rest = new Map();
    root.traverse((o) => {
        if (o.isBone)
            rest.set(o, { p: o.position.clone(), q: o.quaternion.clone() });
    });
    const sides = { Left: 1, Right: -1 };   // 人物面朝 +z 时，左侧在 +x
    const legs = {};
    for (const s of Object.keys(sides)) {
        const hip = world(B[s + 'UpLeg']);
        const knee = world(B[s + 'Leg']);
        const ankle = world(B[s + 'Foot']);
        legs[s] = {
            hip,
            ankle,
            thigh: hip.distanceTo(knee),
            shin: knee.distanceTo(ankle),
            footQ: B[s + 'Foot'].getWorldQuaternion(new THREE.Quaternion()),
        };
    }
    const hipsRest = world(B.Hips);
    const headTop = world(B.HeadTop_End);
    const floorY = Math.min(world(B.LeftToeBase).y, world(B.LeftFoot).y) - 0.02 * (headTop.y - hipsRest.y);

    function resetPose() {
        for (const [b, r] of rest) {
            b.position.copy(r.p);
            b.quaternion.copy(r.q);
        }
        root.updateMatrixWorld(true);
    }

    // 旋转 b，使 b → child 的方向在世界坐标里对准 dir
    function aim(b, child, dir) {
        const cur = world(child).sub(world(b)).normalize();
        q1.setFromUnitVectors(cur, v1.copy(dir).normalize());
        b.getWorldQuaternion(q2);
        q2.premultiply(q1);
        b.parent.getWorldQuaternion(q3);
        b.quaternion.copy(q3.invert().multiply(q2));
        b.updateMatrixWorld(true);
    }

    function setWorldQuaternion(b, q) {
        b.parent.getWorldQuaternion(q3);
        b.quaternion.copy(q3.invert().multiply(q));
        b.updateMatrixWorld(true);
    }

    function moveWorld(b, delta) {
        const target = world(b).add(delta);
        b.parent.updateMatrixWorld(true);
        b.position.copy(b.parent.worldToLocal(target));
        b.updateMatrixWorld(true);
    }

    const lean = (deg) => new THREE.Vector3(0, Math.cos(deg * DEG), Math.sin(deg * DEG));

    // s ∈ [0, 1]：0 站直，1 蹲到底
    function pose(s) {
        resetPose();
        const shinA = 25 * DEG * s;     // 小腿前倾，膝盖向前
        const thighA = 80 * DEG * s;    // 大腿后倒，髋向后坐（80° ≈ 大腿接近水平）
        const kneeOut = 0.05 * s;       // 膝盖略向外，与脚尖同向（米，按 1.7 米身高）
        const scale = (headTop.y - floorY) / 1.7;

        // 先算两条腿的膝、髋位置，再整体平移髋部
        const plan = {};
        const shift = new THREE.Vector3();
        for (const [s2, sign] of Object.entries(sides)) {
            const L = legs[s2];
            const dx = sign * kneeOut * scale;
            const shinLen = Math.sqrt(L.shin * L.shin - dx * dx);
            const thighLen = Math.sqrt(L.thigh * L.thigh - dx * dx);
            const knee = L.ankle.clone().add(new THREE.Vector3(dx, shinLen * Math.cos(shinA), shinLen * Math.sin(shinA)));
            const hip = knee.clone().add(new THREE.Vector3(-dx, thighLen * Math.cos(thighA), -thighLen * Math.sin(thighA)));
            plan[s2] = { knee, hip };
            shift.add(hip.clone().sub(L.hip).multiplyScalar(0.5));
        }
        moveWorld(B.Hips, shift);

        // 躯干前倾，头部前倾得少一些，保持目视前方
        const torso = 48 * s;
        aim(B.Spine, B.Spine1, lean(torso * 0.6));
        aim(B.Spine1, B.Spine2, lean(torso * 0.85));
        aim(B.Spine2, B.Neck, lean(torso));
        aim(B.Neck, B.Head, lean(torso * 0.6));
        aim(B.Head, B.HeadTop_End, lean(torso * 0.2));

        for (const [s2, sign] of Object.entries(sides)) {
            const L = legs[s2];
            const { knee, hip } = plan[s2];
            aim(B[s2 + 'UpLeg'], B[s2 + 'Leg'], knee.clone().sub(hip));
            aim(B[s2 + 'Leg'], B[s2 + 'Foot'], L.ankle.clone().sub(knee));
            setWorldQuaternion(B[s2 + 'Foot'], L.footQ);

            // 手臂：站立时自然下垂略外展，蹲下时前平举配重
            const down = new THREE.Vector3(sign * Math.sin(8 * DEG), -Math.cos(8 * DEG), 0.05);
            const front = new THREE.Vector3(sign * 0.12, 0.05, 1).normalize();
            const arm = down.clone().lerp(front, s).normalize();
            aim(B[s2 + 'Arm'], B[s2 + 'ForeArm'], arm);
            aim(B[s2 + 'ForeArm'], B[s2 + 'Hand'], arm.clone().add(new THREE.Vector3(0, 0.08 * s, 0)));
            aim(B[s2 + 'Hand'], B[s2 + 'HandMiddle1'], arm);
        }
    }

    return { pose, floorY, height: headTop.y - floorY, bones: B, world };
}

// 每次动作的深度曲线：首尾都是站直（0），中间蹲到底（1），所以每次首尾姿态相同
export function squatDepth(phase) {
    return (1 - Math.cos(2 * Math.PI * phase)) / 2;
}
