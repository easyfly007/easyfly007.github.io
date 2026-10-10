# 健身计划生成器：开发约定

- 开始工作前先在仓库根目录 `git pull`，在最新代码上开发
- 进度和下一步看 `PLAN.md`；需求看 `docs/requirements.md`，数据结构和算法看 `docs/design.md`
- 技术栈：React + TypeScript + Vite，测试用 Vitest（界面测试用 jsdom + Testing Library）
- Node 需要 18+。本机系统自带的是 v12，用 nvm 里的 Node 22：`export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"`
- 常用命令（在本目录下）：`npm run dev` / `npm test` / `npx tsc -b` / `npx eslint src` / `npm run build`
- 只做中文
- 生成引擎 `src/engine/` 是纯函数，不依赖界面；改动引擎必须保证 `generate.test.ts` 里"时长误差 < 15%、只用可行动作"的全组合测试通过
- 跟练时长必须精确：所有时长都从 `buildTimeline()` 算出来，不要另写一套估算
- 新动作加在 `src/data/exercises.ts`，标签要填全（器械、场地、鞋、噪音、出汗、强度、受力部位、`secPerRep`）
- TS/TSX 也遵守全局规则：`if`/`for`/`while` 的语句体另起一行，一行只写一条语句
- 每做完一步：测试/类型检查/lint 通过 → 在 `PLAN.md` 更新进度 → `git commit` → `git push` → 等用户检查后再做下一步
- **发布注意**：本目录的 `index.html` 是 Vite 的开发入口，直接推送到 GitHub Pages 打开是空白页。正式发布方案见 `PLAN.md`"发布"一节，定下来之前不要以为推送后页面就能用
