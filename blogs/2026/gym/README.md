# 健身计划生成器

根据你当下的条件（器械、场地、时长、强度、练后能否洗澡等）自动生成一节训练，并可保存为模板下次直接调用。

- 进度与计划：[PLAN.md](PLAN.md)
- 开发约定：[CLAUDE.md](CLAUDE.md)
- 需求：[docs/requirements.md](docs/requirements.md)
- 设计：[docs/design.md](docs/design.md)

## 运行

需要 Node.js 18+（本机用 nvm 的 Node 22：`export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"`）。

```bash
npm install
npm run dev      # 浏览器打开提示的地址
npm test         # 运行测试
```
