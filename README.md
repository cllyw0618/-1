# IntraHC 项目网站

面向 **Intra-hyperedge Contrastive Learning for End-to-End Attributed Hypergraph Clustering（IntraHC）** 的静态展示站点，用于中国大学生计算机设计大赛等场景下的作品说明与交互演示。

## 本地预览

无需构建。将本目录作为网站根目录，用任意静态服务器打开即可，例如：

- 用 VS Code / Cursor 的 **Live Server** 打开 `index.html`，或  
- 在项目根目录执行：`npx serve .`（需已安装 Node.js）

**注意：** 若直接双击用 `file://` 打开，部分浏览器对本地脚本或字体可能有限制，建议始终通过本地 HTTP 访问。

## 主要页面

| 文件 | 说明 |
|------|------|
| `index.html` | 首页 |
| `algorithm.html` | 算法与方法 |
| `results.html` | 实验结果 |
| `workflow.html` | 技术流程 |
| `similar-tech.html` | 同类技术 |
| `resources.html` | 项目资源 |
| `cluster.html` | **立即试用**：超图聚类交互可视化（示例数据已脱敏，页面内有合规说明） |

## 目录结构（简要）

- `css/` — 主题与子页样式（含 `cluster.css` 等）
- `js/` — 交互脚本（含首页 hero、`cluster-visualization.js` 等）
- `assets/` — 图片、图标等静态资源
- `data/cluster/` — 聚类演示用 JSON 与 `cluster-data.js`

## 技术说明

- HTML5 + CSS3，基于 Bootstrap 5 布局与组件  
- 聚类页使用 Canvas 与本地数据驱动可视化  
- 部分页面依赖 CDN（Bootstrap、Font Awesome、Google Fonts），离线环境需自行替换或缓存

## 合规提示

演示数据中人物、机构等以编号代替真实名称，仅作技术展示，详见 `cluster.html` 首页 hero 内的说明。

---

*IntraHC · 网页设计作品仓库*
