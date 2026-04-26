# 项目协作修改指南（TEAM_EDIT_GUIDE）

本项目已从单页改为“首页 + 分页子页”结构。请成员按页面分工修改，避免互相覆盖。

## 1. 先看目录

- `index.html`：首页（保留总览，不要删原板块）
- `algorithm.html`：算法/方法详情页
- `results.html`：实验结果与可视化页
- `workflow.html`：技术流程页
- `resources.html`：资源页（论文/代码/数据/文档）
- `blog.html`：文章列表页
- `blog/intrahc.html`：文章正文示例页
- `css/styles.css`：原主题样式（尽量少动）
- `css/custom-pages.css`：分页新增样式（优先在这里改）
- `assets/`：图片与静态资源

## 2. 每个人该在哪改

### A. 首页总览（`index.html`）

只做“摘要”和“入口按钮”维护，不要把首页改成空壳。

重点位置：
- `id="services"`：算法摘要
- `id="portfolio"`：成果摘要
- `id="about"`：流程摘要
- `id="contact"`：资源/联系入口

如果只想改文案：直接在对应 section 的 `<h2> <h3> <p>` 里改。
如果只想改跳转：修改该 section 里的 `Read More` 按钮链接。

### B. 算法组（`algorithm.html`）

建议维护内容：
- 研究背景
- 方法思路
- 模型结构
- 创新点/技术优势

可改区域：`.feature-card` 卡片内容。

### C. 实验组（`results.html`）

建议维护内容：
- 指标数据（ACC/NMI/ARI/F1）
- 对比实验
- 消融实验
- 可视化图说明

可改区域：`.result-card` 卡片与图片。

### D. 流程组（`workflow.html`）

建议维护内容：
- 步骤描述
- 每一步输入/输出
- 实施路径

可改区域：`.step-card` 和 `.step-index`。

### E. 资源组（`resources.html`）

建议维护内容：
- 论文链接
- 代码仓库链接
- 数据下载链接
- 文档链接

可改区域：`.resource-card`。

### F. 文档/宣传组（`blog.html` + `blog/*.html`）

- 在 `blog.html` 维护文章列表卡片（标题、日期、摘要、链接）
- 每篇正文单独建 `blog/xxx.html`
- 注意子页路径要用 `../css/...`、`../js/...`

## 3. 导航栏怎么改（全员必看）

所有页面都有统一导航。新增页面或重命名页面时：
1. 把每个页面顶部 `<ul class="navbar-nav ...">` 都同步修改
2. 当前页对应链接加 `active`
3. 保证链接是页面路径，不是 `#锚点`

## 4. 样式修改规范

优先改：`css/custom-pages.css`
尽量不改：`css/styles.css`（原主题和已有定制较多）

常用类：
- `.page-hero`
- `.subpage-section`
- `.feature-card`
- `.result-card`
- `.step-card`
- `.resource-card`
- `.article-card`
- `.back-link`
- `.cta-section`

## 5. 图片与资源规范

- 统一放在 `assets/` 下
- 引用路径使用相对路径
- 不要删已有图片
- `blog/` 下页面引用根目录资源时记得加 `../`

## 6. 本地预览

直接双击 `index.html` 打开即可。

推荐检查：
1. 导航是否能跳到对应页面
2. 移动端菜单是否能展开/收起
3. 图片是否 404
4. 当前页面导航高亮是否正确

## 7. 提交前自检清单

- [ ] 没有删除首页主板块
- [ ] 导航链接和 active 正确
- [ ] 子页资源路径正确（尤其 `blog/`）
- [ ] 样式改动主要在 `custom-pages.css`

## 8. 建议分工（避免冲突）

- 成员 A：`algorithm.html`
- 成员 B：`results.html`
- 成员 C：`workflow.html`
- 成员 D：`resources.html` + `blog.html`
- 负责人：只在最后统一合并 `index.html` 导航与摘要按钮

---

如果你不知道从哪开始：
先改自己负责的子页，再回到 `index.html` 对应板块把摘要和按钮同步一下。
