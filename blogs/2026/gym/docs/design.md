# 设计文档

## 技术栈

- React + TypeScript + Vite
- 测试：Vitest
- 存储：浏览器 `localStorage`（通过 `storage` 模块封装，以后可换 IndexedDB 或后端）
- 动作库：`src/data/exercises.ts`，纯数据，方便查看和修改

## 目录结构

```
src/
  types.ts            核心类型
  data/exercises.ts   动作库
  engine/             生成引擎（纯函数，无 UI 依赖）
    filter.ts         按条件过滤动作
    generate.ts       组合成一节训练
    random.ts         可复现的伪随机数
    timeline.ts       把训练展开成跟练时间线（每一步精确到秒）
  storage/            模板保存/读取
  components/         UI 组件
```

## 数据结构

### 训练条件 `Conditions`

```ts
{
  equipment: Equipment[]      // 拥有的器械（徒手不需要列出）
  location: 'indoor' | 'outdoor'
  space: 'small' | 'large'
  quiet: boolean              // 需要安静
  hasShoes: boolean
  durationMin: number         // 15 / 20 / 30 / 45 / 60
  intensity: 1 | 2 | 3        // 低 / 中 / 高
  canShower: boolean
  goal: 'general' | 'fat_loss' | 'strength' | 'flexibility'
  limitations: Limitation[]   // 'knee' | 'back' | 'shoulder' | 'wrist'
  rest?: 'short' | 'normal' | 'long'  // 休息档位，缺省为 normal
}
```

### 动作 `Exercise`

```ts
{
  id: string
  name: string                // 中文名
  equipment: Equipment[]      // 全部需要；空数组 = 徒手
  locations: Location[]       // 适用场地
  needsShoes: boolean
  needsSpace: boolean         // 需要移动空间
  noisy: boolean              // 跳跃等有噪音
  phases: Phase[]             // 可用于 'warmup' | 'main' | 'cooldown'
  category: 'strength' | 'cardio' | 'core' | 'mobility' | 'stretch'
  muscles: Muscle[]
  sweat: 1..5                 // 出汗等级
  intensity: 1..5             // 强度等级
  stresses: Limitation[]      // 对哪些部位压力大
  mode: 'reps' | 'time'       // 计次 / 计时
  defaultReps?: number
  defaultSeconds?: number
  secPerRep: number           // 计次动作每次耗时（秒），用于换算跟练时长
  perSide?: boolean           // 左右各做一次
  cues: string[]              // 动作要点
}
```

### 训练 `Workout`

```ts
{
  seed: number                // 随机种子，同条件+同种子 = 同结果
  conditions: Conditions
  blocks: Block[]
  estimatedMin: number
}

Block {
  phase: Phase
  title: string
  format: 'straight' | 'circuit' | 'flow'
  rounds: number              // 循环轮数（straight 为 1）
  restBetweenRoundsSec: number
  items: WorkoutItem[]
}

WorkoutItem {
  exerciseId: string
  sets: number                // 组数（circuit/flow 为 1，轮数在 Block 上）
  reps?: number
  seconds?: number
  restSec: number             // 每组/每个动作后的休息
}
```

### 模板 `SavedWorkout`

```ts
{
  id: string
  name: string
  workout: Workout            // 包含 conditions
  createdAt: string           // ISO 时间
  lastUsedAt?: string
}
```

## 生成算法

1. **过滤**：按器械、场地、空间、安静、鞋、身体限制、出汗上限过滤出合法动作。
   - 出汗上限：不能洗澡 → 2；能洗澡 → 低强度 3、中 4、高 5
   - 强度上限：低 2、中 4、高 5
2. **时间分配**：热身 ≈ 15%，放松 ≈ 15%，其余给主训练。
3. **选择形式**：
   - 力量目标，或不能洗澡 → `straight`（常规组，较长休息）
   - 减脂，或综合且强度高 → `circuit`（计时循环）
   - 柔韧放松 → `flow`（拉伸流）
   - 其余 → `circuit`，强度决定做/休比例
4. **选动作**：可复现随机数 + 目标偏好（减脂偏有氧、力量偏负重、按强度偏高/低强度动作）
   + 优先覆盖还没练到的肌群，排出候选顺序。
5. **按时长填充**（时长都用跟练时间线精确计算，保证预估 = 实际）：
   - 循环：在 每轮 3–6 个动作 × 1–10 轮 中选最接近目标的组合
   - 常规组：在 动作数 × 组数（2–5）中选最接近目标的组合，偏向该强度的默认组数
   - 休息档位：主训练的组间/动作间/轮间休息在默认值上 −15 / 0 / +15 秒，最少 10 秒；热身、放松、拉伸流的 5 秒衔接不变
   - 拉伸流/热身/放松：逐个追加动作直到不再更接近目标；动作不够时把停留时间最多拉长到 2 倍，再从头循环

## 跟练时间线

训练展开为一串步骤 `Step[]`，每步有确定的秒数，跟练播放器只需按顺序倒计时：

```ts
Step {
  kind: 'prepare' | 'work' | 'rest'
  exerciseId?: string         // work 当前动作；rest/prepare 为下一个动作（用于预告）
  seconds: number
  label: string               // 屏幕和语音展示的文字，如"卷腹 15 次"
  side?: 'left' | 'right'     // 单侧动作拆为左右两步
}
```

整节课总时长 = 所有步骤秒数之和，与生成时的预估时长一致。

## 训练记录 `WorkoutRecord`

```ts
{
  id: string
  workout: Workout
  name?: string               // 从模板开练时的模板名
  startedAt: string           // ISO
  endedAt: string
  played: [number, number][]  // 实际播放过的时间线区间（秒）；点"跳过"时结束当前区间
}
```

只存原始数据，报告里的数字都在 `src/engine/report.ts` 里由 `buildTimeline()` 和 `played` 重新算出：

- 每步完成秒数 = 该步时间范围与 `played` 的重叠；做到 90% 算这一步完成；单侧动作左右两步合起来算一组
- 计次动作完成次数 = 完成秒数 ÷ `secPerRep`（向下取整）
- 估算热量：千卡 = MET × 3.5 × 体重(kg) ÷ 200 × 分钟；动作 MET 按强度等级 1..5 取 3 / 4 / 5.5 / 7 / 8.5，休息取 1.5
- 按本地日期汇总每天、每周（周一开始）的时长与次数，以及连续打卡天数
- 存在 localStorage `gym.history.v1`（最多 1000 条），体重存在 `gym.weightKg.v1`

## 动作动画（`src/anim/`）

- `types.ts`：姿态 `Pose`（身体朝向 pitch/roll/yaw、离地 lift、躯干/头/上臂/前臂/大腿/小腿方向向量、脚尖方向）与动画 `AnimDef`（关键帧、锚点、器械）
- `pose.ts`：写姿态的辅助函数（`limb`/`up`/`legs`/`arms` 等，用"前抬几度、外展几度、弯几度"描述）、关键帧插值（缓入缓出）、左右镜像
- `figure.ts`：three.js 人偶。每段肢体按方向摆放；摆好后自动贴地（最低点落地），`anchor` 指定的部位水平不动（脚不滑步）；`hang` 时手固定在单杠高度；器械按手/肩位置每帧摆放；`plant` 让一只脚踩在凳面高度（登台阶）；姿态里的 `dx`/`dz` 整体平移（滑冰跳）；`repsPerCycle: 2` 表示一个循环含左右各一次
- 写姿态的注意点：身体整体前倾（`pitch`）时，相对骨盆写的腿会跟着转，硬拉、划船、弓步拉伸这类动作的腿要用 `legsWorld` 按世界坐标写；同一个动作的各关键帧 `armsWorld`/`legsWorld` 要一致
- `library.ts`：每个动作的关键帧。单侧动作只写左侧
- `schedule.ts`：一个循环的秒数（计次 = secPerRep）和当前视角（循环交界处切换，每个视角至少 2.5 秒）
- `stage.ts`：场景、灯光、地面、瑜伽垫、三个机位（按整个循环的身体范围自动取景）；由 `ExerciseAnimation` 组件按需加载（three.js 单独打包）
- 检查：`npx vite-node scripts/anim-preview.ts <目录> [动作 id ...]` 输出每个动作 4 个时刻的正面/侧面骨架图（SVG）；`anim.test.ts` 检查贴地和锚点不滑动

## 手动调整训练（`src/engine/edit.ts`）

- `addExercise(workout, id)`：按动作的 `phases` 选区块（main > warmup > cooldown，缺区块就新建）；新动作的组数、次数/时长、休息跟随区块第一个动作（循环区块用计时做/休）；`estimatedMin` 由时间线重算
- `removeItem(workout, blockIndex, itemIndex)`：移除后丢掉空区块，重算时长
- 喜欢的动作：`generateWorkout(c, seed, favorites)`，在排序打分里加 0.7（所有阶段），只影响先后，不绕过过滤；存在 localStorage `gym.favorites.v1`
