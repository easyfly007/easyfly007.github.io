# 用 Mixamo 素材替换火柴人：流程

目标：保留现在的 three.js 舞台（三个机位、按次换视角、时长精确对齐 `secPerRep`、女男教练、深色模式），
只把几何体拼的人偶换成带骨骼的真人风格模型，用 Mixamo 的动作捕捉动画驱动。

整条线：**Mixamo 网站 → FBX 文件 → 预览页标起止帧 → Blender 脚本转换 → GLB 文件 → 播放器加载**。

## 目前进度

- [x] 预览页 `demo/video-mixamo.html`：直接加载 FBX，三个机位看效果，标"一次动作"的起止帧，输出 CSV 行
- [ ] 用户确认风格（人物、动作）
- [ ] `docs/anim-clips.csv`：每个动作的起止帧
- [ ] `scripts/mixamo-to-glb.py`：Blender 批量转换，输出 `public/models/<coach>.glb`
- [ ] 播放器加载 GLB、AnimationMixer 驱动、器械挂手骨、没有的动作退回火柴人

## 素材目录（不提交 git）

```
assets/mixamo/characters/female.fbx   人物，Mixamo 下载时选 With Skin，T-pose，约 50 MB
assets/mixamo/characters/male.fbx
assets/mixamo/anims/<动作 id>.fbx      动作，Without Skin，30 fps，Keyframe Reduction = none，
                                      有 In Place 勾选框就勾上；文件名用 src/data/exercises.ts 里的 id
```

Mixamo 下载选项：Format = FBX Binary（不要 Unity / Unreal 变体）。

## 预览页用法

```
npm run dev
http://localhost:5173/demo/video-mixamo.html
```

1. 选人物和动作，人物第一次加载约 50 MB 要等几秒
2. 拖进度条找一次动作的开始帧，点"设为起点"；找结束帧，点"设为终点"。要求起止姿态一致，循环播放看不出跳动
3. "按每次时长播放"填动作库里的 `secPerRep`，看缩放后快慢是否合适
4. 单侧动作勾"左右镜像"看另一侧
5. 复制 CSV 行，追加到 `docs/anim-clips.csv`

CSV 列：`动作 id, 文件名, 起点帧, 终点帧, 是否镜像`

## 已知情况

- 两个人物文件的骨骼名前缀不同（`mixamorig`、`mixamorig5`），动作文件统一是 `mixamorig`；预览页按去掉前缀后的名字对应，正式转换脚本也要这么做
- 人物文件里 Hips 下面有同名的空骨骼，是 Mixamo 导出的正常现象，不影响驱动
- Mixamo 单位是厘米，加载后整体缩放 0.01
- 动作文件通常是连做几次或带起手收尾，所以必须标起止帧；播放器一段 = 一次

## 后续：Mixamo 没有的动作

手机录真人做一次 → 视频转动捕服务（Rokoko Vision、DeepMotion 等）导出 Mixamo 骨架的 FBX →
放进 `assets/mixamo/anims/` → 同样走预览页标帧和转换脚本。
