# 烟霾季 2026 页面：开发约定

- 计划和进度看 `PLAN.md`，资料和来源链接看 `RESEARCH.md`
- 页面只有一个文件 `index.html`：一个完整的 HTML 文档，CSS 和 JS 都内联，不用构建工具。图形用 SVG/Canvas。外部资源只有 Google Fonts 和首屏实时 PSI 接口（data.gov.sg）
- 图片：只放能合法转载的（如 NASA 卫星图，注明来源），存在 `img/` 里，必须标注时间、来源、地点。新闻实拍照片有版权，不下载、不外链，只做"链接卡片"（时间、摄影/机构、地点、图注译文，点击到原网页看）
- 发布：提交到 `easyfly007.github.io` 仓库的 `master` 并推送，GitHub Pages 自动更新 https://easyfly007.github.io/blogs/2026/haze/
- 只做中文
- **每个数字或事件后面都紧跟来源链接**：`<a class="src" href="…" target="_blank" rel="noopener">简称</a>`（会显示成 [简称↗]）。JS 里生成的内容用 `srcLinks()`。找不到出处的数字不放进页面；新资料先写进 `RESEARCH.md`
- 颜色只用 `:root` 里的 token；深色模式要在两个 dark 块里同步改；PSI 五级色是唯一的语义色
- JS 也遵守全局规则：`if`/`for` 的语句体另起一行，一行只写一条语句
- 每做完一步：在 `PLAN.md` 更新进度 → `git commit` → `git push` → 等用户检查后再做下一步
- 改完后用 `node --check` 检查内联脚本的语法（把 `<script>` 内容提取出来再检查）
