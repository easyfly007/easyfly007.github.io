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
4. **选动作**：用可复现随机数打乱候选，优先覆盖不同肌群，避免重复。
5. **按时长填充**：逐个估算每个动作耗时，填满目标时长为止。

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
