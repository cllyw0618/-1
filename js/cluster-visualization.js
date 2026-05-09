window.addEventListener("DOMContentLoaded", () => {
    const APPLICATION_SCENARIOS = [
        {
            id: "academic",
            title: "学术知识网络",
            subtitle: "相关论文发现 / 共引关系挖掘 / 学术主题聚类",
            datasets: ["Cora", "CiteSeer", "PubMed"],
            defaultDataset: "Cora",
            description: "适用于论文、知识条目和引用关系构成的高阶网络分析。",
        },
        {
            id: "collaboration",
            title: "科研合作网络",
            subtitle: "科研合作群体发现 / 共作者关系挖掘 / 潜在合作推荐",
            datasets: ["DBLP"],
            defaultDataset: "DBLP",
            description: "适用于作者、团队和论文合作关系构成的高阶协作网络。",
        },
        {
            id: "news",
            title: "新闻话题网络",
            subtitle: "新闻话题聚合 / 帖子自动归类 / 舆情主题发现",
            datasets: ["20newsW100"],
            defaultDataset: "20newsW100",
            description: "适用于新闻、帖子和主题共现关系形成的文本高阶关系网络。",
        },
        {
            id: "small-demo",
            title: "小型属性网络",
            subtitle: "小规模属性聚类 / 教学演示 / 可解释性展示",
            datasets: ["Zoo"],
            defaultDataset: "Zoo",
            description: "适用于小规模属性数据的直观演示，便于观察节点、超边和簇之间的关系。",
        },
        {
            id: "multimodal",
            title: "多模态关系网络",
            subtitle: "多模态样本组织 / 行为类别发现 / 关系结构探索",
            datasets: ["Mushroom", "NTU2012"],
            defaultDataset: "Mushroom",
            description: "可作为补充展示，不建议作为页面默认主案例。",
        },
    ];

    const DATASET_ALIASES = {
        cora: "cora",
        citeseer: "citeseer",
        pubmed: "pubmed",
        dblp: "cora_coauthor",
        "20newsw100": "20newsw100",
        zoo: "zoo",
        mushroom: "mushroom",
        ntu2012: "ntu2012",
    };

    const DATASET_SUMMARY_TEXT = {
        cora: "当前展示学术知识网络样例，系统将论文或知识节点抽象为匿名节点，并基于引用、共现或高阶关联结构发现潜在主题群体。",
        citeseer: "当前展示学术知识网络样例，系统可用于共引关系聚类与研究主题结构分析。",
        pubmed: "当前展示学术知识网络样例，系统可用于生物医学文献主题群体发现。",
        cora_coauthor: "当前展示科研合作网络样例，系统用于发现潜在合作群体、共作者社区和高阶协作关系。",
        "20newsw100": "当前展示新闻话题网络样例，系统用于聚合文本主题、发现论坛话题群体和潜在舆情结构。",
        zoo: "当前展示小型属性网络样例，适合观察属性相似性与高阶关系共同作用下的聚类结果。",
        mushroom: "当前展示多模态关系网络补充样例，可用于关系结构探索。",
        ntu2012: "当前展示多模态关系网络补充样例，可用于行为类别关系结构分析。",
    };

    const clusterCanvas = document.getElementById("clusterCanvas");
    if (!clusterCanvas) return;
    const ctx = clusterCanvas.getContext("2d");
    const CANVAS_CAPTION_HEIGHT = 44;
    const CANVAS_CAPTION_TEXT = "整体聚类视图：点击节点查看局部解释；超边用于表示多节点共同关系。";

    const scenarioGrid = document.getElementById("applicationScenarioGrid");
    const scenarioSelect = document.getElementById("scenarioSelect");
    const datasetSelect = document.getElementById("datasetSelect");
    const colorMode = document.getElementById("colorMode");
    const edgeFilter = document.getElementById("edgeFilter");
    const showHyperedges = document.getElementById("showHyperedges");
    const analysisTabs = document.getElementById("analysisViewTabs");
    const stepTabs = document.getElementById("stepViewButtons");
    const searchInput = document.getElementById("nodeSearch");
    const searchButton = document.getElementById("searchNode");
    const relayoutButton = document.getElementById("relayoutView");
    const resetButton = document.getElementById("resetFocus");
    const exportButton = document.getElementById("exportView");
    const startButton = document.getElementById("startAnalysisBtn");
    const viewScenarioBtn = document.getElementById("viewScenarioBtn");
    const analysisMain = document.getElementById("analysisMain");
    const scenarioSection = document.getElementById("scenarioSection");

    const stageLabel = document.getElementById("stageLabel");
    const sampleDatasetLabel = document.getElementById("sampleDatasetLabel");
    const nodeCount = document.getElementById("nodeCount");
    const clusterCount = document.getElementById("clusterCount");
    const hyperedgeCount = document.getElementById("hyperedgeCount");
    const heroAcc = document.getElementById("heroAcc");
    const heroNmi = document.getElementById("heroNmi");
    const datasetSummaryCard = document.getElementById("datasetSummaryCard");
    const datasetSummary = document.getElementById("datasetSummary");
    const hyperedgeExplain = document.getElementById("hyperedgeExplain");
    const dataStatus = document.getElementById("dataStatus");
    const analysisNarrative = document.getElementById("analysisNarrative");

    const selectionPanel = document.getElementById("selectionPanel");
    const recommendationPanel = document.getElementById("recommendationPanel");
    const recommendReasonPanel = document.getElementById("recommendReasonPanel");
    const metricPanel = document.getElementById("metricPanel");
    const hyperedgePanel = document.getElementById("hyperedgePanel");
    const clusterMatrix = document.getElementById("clusterMatrix");
    const topEdgeList = document.getElementById("topEdgeList");
    const crossEdgeList = document.getElementById("crossEdgeList");
    const boundaryNodeList = document.getElementById("boundaryNodeList");
    const maxHyperedgeList = document.getElementById("maxHyperedgeList");
    const highConfidenceClusterList = document.getElementById("highConfidenceClusterList");
    const clusterTooltip = document.getElementById("clusterTooltip");
    const canvasEmpty = document.getElementById("canvasEmpty");
    const canvasLoading = document.getElementById("canvasLoading");

    const clusterPalette = ["#246bfe", "#00a884", "#f59e0b", "#ef4444", "#7c3aed", "#0891b2", "#65a30d", "#db2777", "#475569", "#ea580c"];

    const state = {
        manifest: null,
        availableDatasetMap: new Map(),
        currentScenarioId: "academic",
        currentDatasetFile: null,
        currentDatasetName: null,
        currentView: "global",
        colorMode: "cluster",
        edgeFilter: "all",
        showHyperedges: false,
        selectedNodeId: null,
        hoveredNodeId: null,
        quickMode: null,
        loading: false,
        nodes: [],
        nodesById: new Map(),
        hyperedges: [],
        hyperedgesById: new Map(),
        metrics: {},
        stats: {},
        clusterStats: [],
        localLayoutCache: new Map(),
        globalLayoutCache: null,
        hitPoints: [],
        viewTransform: {
            scale: 1,
            panX: 0,
            panY: 0,
        },
        pointer: {
            mode: null,
            nodeId: null,
            edgeId: null,
            moved: false,
            last: null,
        },
        activeLayoutRef: null,
        loadToken: 0,
    };

    function canon(value) {
        return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    }

    function clamp(v, min, max) {
        return Math.max(min, Math.min(max, v));
    }

    function safeNumber(v) {
        const n = Number(v);
        return Number.isFinite(n) ? n : null;
    }

    function setStatus(msg) {
        if (dataStatus) dataStatus.textContent = msg;
    }

    function setNarrative(msg) {
        if (analysisNarrative) analysisNarrative.textContent = msg;
    }

    function nextPaint() {
        return new Promise((resolve) => {
            if (typeof window.requestAnimationFrame === "function") {
                window.requestAnimationFrame(() => resolve());
            } else {
                setTimeout(resolve, 0);
            }
        });
    }

    function syncHyperedgeExplain() {
        if (hyperedgeExplain) hyperedgeExplain.hidden = !state.showHyperedges;
    }

    function setLoading(flag, msg = "") {
        state.loading = flag;
        canvasLoading?.classList.toggle("hidden", !flag);
        if (flag && msg) setStatus(msg);
    }

    function showEmpty(text) {
        if (!canvasEmpty) return;
        canvasEmpty.classList.remove("hidden");
        canvasEmpty.textContent = text;
    }

    function hideEmpty() {
        canvasEmpty?.classList.add("hidden");
    }

    function resizeCanvas() {
        const rect = clusterCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        clusterCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
        clusterCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        render();
    }

    function getCanvasPoint(event) {
        const rect = clusterCanvas.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function clearCanvas() {
        const w = clusterCanvas.clientWidth;
        const h = clusterCanvas.clientHeight;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(15,23,42,0.045)";
        ctx.lineWidth = 1;
        for (let x = 42; x < w; x += 42) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 42; y < h; y += 42) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    }

    function colorForNode(node) {
        if (state.colorMode === "purity") {
            if (node.boundaryScore >= 0.6) return "#ef4444";
            if (node.boundaryScore >= 0.35) return "#f59e0b";
            return "#16a34a";
        }
        return clusterPalette[Math.abs(node.cluster) % clusterPalette.length];
    }

    function edgeColor(edge) {
        if (edge.purity >= 0.8) return "#16a34a";
        if (edge.purity >= 0.55) return "#f59e0b";
        return "#ef4444";
    }

    function getEmbeddingPosition(nodeRaw) {
        if (typeof nodeRaw.x === "number" && typeof nodeRaw.y === "number") return { x: nodeRaw.x, y: nodeRaw.y };
        if (Array.isArray(nodeRaw.embed2d) && nodeRaw.embed2d.length >= 2) return { x: nodeRaw.embed2d[0], y: nodeRaw.embed2d[1] };
        if (Array.isArray(nodeRaw.embedding_2d) && nodeRaw.embedding_2d.length >= 2) return { x: nodeRaw.embedding_2d[0], y: nodeRaw.embedding_2d[1] };
        if (Array.isArray(nodeRaw.coords) && nodeRaw.coords.length >= 2) return { x: nodeRaw.coords[0], y: nodeRaw.coords[1] };
        if (Array.isArray(nodeRaw.position) && nodeRaw.position.length >= 2) return { x: nodeRaw.position[0], y: nodeRaw.position[1] };
        return null;
    }

    function normalizeEmbeds(nodes) {
        const valids = nodes
            .map((node) => ({ node, raw: getEmbeddingPosition(node.raw) }))
            .filter((item) => item.raw && Number.isFinite(item.raw.x) && Number.isFinite(item.raw.y));
        if (!valids.length) {
            const cols = Math.ceil(Math.sqrt(nodes.length || 1));
            nodes.forEach((node, index) => {
                const row = Math.floor(index / cols);
                const col = index % cols;
                node.embed = {
                    x: cols <= 1 ? 0.5 : col / (cols - 1),
                    y: cols <= 1 ? 0.5 : row / (cols - 1),
                    fallback: true,
                };
            });
            return;
        }
        const xs = valids.map((item) => Number(item.raw.x));
        const ys = valids.map((item) => Number(item.raw.y));
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        const in01 = minX >= 0 && maxX <= 1 && minY >= 0 && maxY <= 1;
        const in100 = minX >= 0 && maxX <= 100 && minY >= 0 && maxY <= 100;
        const dx = maxX - minX || 1;
        const dy = maxY - minY || 1;

        nodes.forEach((node, idx) => {
            const raw = getEmbeddingPosition(node.raw);
            if (!raw) {
                const cols = Math.ceil(Math.sqrt(nodes.length || 1));
                const row = Math.floor(idx / cols);
                const col = idx % cols;
                node.embed = {
                    x: cols <= 1 ? 0.5 : col / (cols - 1),
                    y: cols <= 1 ? 0.5 : row / (cols - 1),
                    fallback: true,
                };
                return;
            }
            let nx = Number(raw.x);
            let ny = Number(raw.y);
            if (in01) {
                node.embed = { x: nx, y: ny, fallback: false };
            } else if (in100) {
                node.embed = { x: nx / 100, y: ny / 100, fallback: false };
            } else {
                node.embed = { x: (nx - minX) / dx, y: (ny - minY) / dy, fallback: false };
            }
        });
    }

    function parseRecommendArray(input, defaultReason, nodeMap) {
        if (!Array.isArray(input)) return [];
        return input
            .map((item) => {
                if (typeof item === "number") {
                    const n = nodeMap.get(item);
                    if (!n) return null;
                    return {
                        id: n.id,
                        cluster: n.cluster,
                        score: 0.5,
                        shared_hyperedges: 0,
                        reason: defaultReason,
                    };
                }
                if (!item || typeof item !== "object") return null;
                const id = Number.isFinite(Number(item.id)) ? Number(item.id) : Number(item.node_id);
                if (!Number.isFinite(id)) return null;
                const n = nodeMap.get(id);
                return {
                    id,
                    cluster: Number.isFinite(Number(item.cluster)) ? Number(item.cluster) : n?.cluster ?? 0,
                    score: Number.isFinite(Number(item.score)) ? Number(item.score) : 0.5,
                    shared_hyperedges: Number.isFinite(Number(item.shared_hyperedges)) ? Number(item.shared_hyperedges) : 0,
                    reason: item.reason || defaultReason,
                };
            })
            .filter(Boolean);
    }

    function cosine(a, b) {
        const dot = a.x * b.x + a.y * b.y;
        const na = Math.hypot(a.x, a.y) || 1;
        const nb = Math.hypot(b.x, b.y) || 1;
        return (dot / (na * nb) + 1) / 2;
    }

    function preprocessDataset(raw) {
        const nodes = (raw.nodes || []).map((item, index) => ({
            id: Number.isFinite(Number(item.id)) ? Number(item.id) : index,
            raw: item,
            cluster: Number.isFinite(Number(item.cluster)) ? Number(item.cluster) : 0,
            label: Number.isFinite(Number(item.label)) ? Number(item.label) : null,
            confidence: Number.isFinite(Number(item.confidence)) ? clamp(Number(item.confidence), 0, 1) : null,
            embed: null,
            edgeIds: [],
            incident_edge_ids: [],
            top_similar: [],
            hyperedge_neighbors: [],
            boundaryScore: 0,
            crossClusterNeighborRatio: 0,
            lowPurityEdgeCount: 0,
        }));
        const nodeMap = new Map(nodes.map((n) => [n.id, n]));
        normalizeEmbeds(nodes);

        const hyperedges = (raw.hyperedges || [])
            .map((edge, index) => {
                const refs = Array.isArray(edge.nodes) ? edge.nodes : [];
                const members = refs
                    .map((ref) => {
                        if (nodeMap.has(ref)) return nodeMap.get(ref);
                        if (Number.isInteger(ref) && ref >= 0 && ref < nodes.length) return nodes[ref];
                        return null;
                    })
                    .filter(Boolean);
                if (members.length < 2) return null;
                const clusterCount = new Map();
                members.forEach((n) => clusterCount.set(n.cluster, (clusterCount.get(n.cluster) || 0) + 1));
                const sorted = [...clusterCount.entries()].sort((a, b) => b[1] - a[1]);
                const purityRaw = safeNumber(edge.purity);
                const purity = purityRaw === null ? sorted[0][1] / members.length : clamp(purityRaw, 0, 1);
                const id = Number.isFinite(Number(edge.id)) ? Number(edge.id) : index;
                return {
                    id,
                    members,
                    memberIds: members.map((m) => m.id),
                    size: members.length,
                    purity,
                    dominantCluster: sorted[0][0],
                    clusters: sorted.map(([cluster, count]) => ({ cluster, count })),
                    mixed: sorted.length > 1,
                };
            })
            .filter(Boolean);
        const edgeMap = new Map(hyperedges.map((e) => [e.id, e]));

        hyperedges.forEach((edge) => {
            edge.members.forEach((node) => {
                node.edgeIds.push(edge.id);
                node.incident_edge_ids.push(edge.id);
            });
        });

        const sharedMap = new Map(nodes.map((n) => [n.id, new Map()]));
        hyperedges.forEach((edge) => {
            edge.members.forEach((a) => {
                edge.members.forEach((b) => {
                    if (a.id === b.id) return;
                    const map = sharedMap.get(a.id);
                    map.set(b.id, (map.get(b.id) || 0) + 1);
                });
            });
        });

        nodes.forEach((node) => {
            const incidentEdges = node.edgeIds.map((id) => edgeMap.get(id)).filter(Boolean);
            if (node.confidence === null) {
                if (!incidentEdges.length) node.confidence = 0.5;
                else {
                    const avgPurity = incidentEdges.reduce((sum, e) => sum + e.purity, 0) / incidentEdges.length;
                    const ownClusterRatio = incidentEdges.filter((e) => e.dominantCluster === node.cluster).length / incidentEdges.length;
                    node.confidence = clamp(avgPurity * 0.5 + ownClusterRatio * 0.5, 0.22, 0.98);
                }
            }
            node.top_similar = parseRecommendArray(node.raw.top_similar, "同簇且嵌入距离较近，说明模型认为二者属性和局部结构相似。", nodeMap);
            if (!node.top_similar.length) {
                node.top_similar = nodes
                    .filter((n) => n.id !== node.id && n.cluster === node.cluster)
                    .map((n) => ({
                        id: n.id,
                        cluster: n.cluster,
                        score: Number(cosine(node.embed, n.embed).toFixed(3)),
                        shared_hyperedges: 0,
                        reason: "同簇且嵌入距离较近，说明模型认为二者属性和局部结构相似。",
                    }))
                    .sort((a, b) => b.score - a.score)
                    .slice(0, 10);
            }
            node.hyperedge_neighbors = parseRecommendArray(node.raw.hyperedge_neighbors, "与当前节点共享高阶超边，说明二者共同出现在同一组关系中，可作为结构相关性的依据。", nodeMap);
            if (!node.hyperedge_neighbors.length) {
                node.hyperedge_neighbors = [...sharedMap.get(node.id).entries()]
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 10)
                    .map(([id, count]) => {
                        const n = nodeMap.get(id);
                        return {
                            id,
                            cluster: n?.cluster ?? 0,
                            score: 0.5,
                            shared_hyperedges: count,
                            reason:
                                count >= 2
                                    ? "与当前节点共享多条高阶超边，说明二者反复出现在同类群体关系中，结构相关性更强。"
                                    : "与当前节点共享高阶超边，说明二者共同出现在同一组关系中，可作为结构相关性的依据。",
                        };
                    });
            }

            const neighbors = new Set();
            const crossNeighbors = new Set();
            let lowPurityEdgeCount = 0;
            incidentEdges.forEach((edge) => {
                if (edge.purity < 0.55) lowPurityEdgeCount += 1;
                edge.members.forEach((m) => {
                    if (m.id === node.id) return;
                    neighbors.add(m.id);
                    if (m.cluster !== node.cluster) crossNeighbors.add(m.id);
                });
            });
            node.lowPurityEdgeCount = lowPurityEdgeCount;
            node.crossClusterNeighborRatio = neighbors.size ? crossNeighbors.size / neighbors.size : 0;
            const lowPurityRatio = incidentEdges.length ? lowPurityEdgeCount / incidentEdges.length : 0;
            node.boundaryScore = clamp(lowPurityRatio * 0.45 + node.crossClusterNeighborRatio * 0.35 + (1 - node.confidence) * 0.2, 0, 1);
        });

        const clusterGroups = new Map();
        nodes.forEach((node) => {
            if (!clusterGroups.has(node.cluster)) clusterGroups.set(node.cluster, []);
            clusterGroups.get(node.cluster).push(node);
        });
        const clusterStats = [...clusterGroups.entries()]
            .map(([clusterId, members]) => {
                const avgConfidence = members.reduce((sum, n) => sum + n.confidence, 0) / Math.max(1, members.length);
                const avgBoundary = members.reduce((sum, n) => sum + n.boundaryScore, 0) / Math.max(1, members.length);
                return {
                    clusterId,
                    size: members.length,
                    avgConfidence,
                    consistency: clamp(avgConfidence * (1 - avgBoundary), 0, 1),
                };
            })
            .sort((a, b) => b.consistency - a.consistency);

        return {
            dataset: raw.dataset || "Unknown",
            nodes,
            nodesById: nodeMap,
            hyperedges,
            hyperedgesById: edgeMap,
            metrics: raw.metrics || {},
            stats: {
                nodeCount: Number(raw.node_count || nodes.length),
                shownNodeCount: Number(raw.shown_node_count || nodes.length),
                clusterCount: Number(raw.cluster_count || clusterStats.length),
                hyperedgeCount: hyperedges.length,
            },
            clusterStats,
        };
    }

    function getCurrentScenario() {
        return APPLICATION_SCENARIOS.find((s) => s.id === state.currentScenarioId) || APPLICATION_SCENARIOS[0];
    }

    function getDatasetEntryByLabel(label) {
        const key = canon(label);
        const aliased = DATASET_ALIASES[key] || key;
        return state.availableDatasetMap.get(aliased) || null;
    }

    function getAvailableScenarioDatasetEntries(scenario) {
        return scenario.datasets
            .map((label) => {
                const entry = getDatasetEntryByLabel(label);
                return { label, entry };
            })
            .filter((item) => item.entry);
    }

    function getDisplayDatasetLabel(file) {
        const scenario = getCurrentScenario();
        const entries = getAvailableScenarioDatasetEntries(scenario);
        const found = entries.find((item) => item.entry.file === file);
        if (found) return found.label;
        const manifestEntry = (state.manifest?.datasets || []).find((item) => item.file === file);
        return manifestEntry?.name || file.replace(".json", "");
    }

    function renderScenarioCards() {
        if (!scenarioGrid) return;
        const html = APPLICATION_SCENARIOS.map((scenario) => {
            const activeScenario = state.currentScenarioId === scenario.id;
            const activeDatasetLabel = activeScenario && state.currentDatasetFile ? getDisplayDatasetLabel(state.currentDatasetFile) : scenario.defaultDataset;
            const enterDisabled = getDatasetEntryByLabel(activeDatasetLabel) ? "" : "disabled";
            const datasetButtons = scenario.datasets
                .map((label) => {
                    const entry = getDatasetEntryByLabel(label);
                    const disabled = entry ? "" : "disabled";
                    const active = activeScenario && entry && state.currentDatasetFile === entry.file ? "active" : "";
                    const pressed = active ? "true" : "false";
                    return `<button type="button" data-scenario-id="${scenario.id}" data-dataset-label="${label}" class="${active}" aria-pressed="${pressed}" ${disabled}>${label}</button>`;
                })
                .join("");
            return `<article class="scenario-card ${activeScenario ? "active" : ""}">
                <h3>${scenario.title}</h3>
                <p>${scenario.description}</p>
                <p>可选数据：${scenario.datasets.join(" / ")}</p>
                <div class="scenario-datasets">${datasetButtons}</div>
                <button type="button" class="enter-btn" data-enter-scenario="${scenario.id}" data-enter-dataset-label="${activeDatasetLabel}" ${enterDisabled}>加载该场景</button>
            </article>`;
        }).join("");
        scenarioGrid.innerHTML = html;
    }

    function updateScenarioSelects() {
        if (scenarioSelect) {
            scenarioSelect.innerHTML = APPLICATION_SCENARIOS.map((s) => `<option value="${s.id}">${s.title}</option>`).join("");
            scenarioSelect.value = state.currentScenarioId;
        }
        const scenario = getCurrentScenario();
        const entries = getAvailableScenarioDatasetEntries(scenario);
        if (datasetSelect) {
            datasetSelect.innerHTML = entries.map((item) => `<option value="${item.entry.file}">${item.label}</option>`).join("");
            if (!entries.find((item) => item.entry.file === state.currentDatasetFile)) {
                datasetSelect.value = entries[0]?.entry.file || "";
            } else {
                datasetSelect.value = state.currentDatasetFile;
            }
        }
    }

    function formatScenarioSummary(datasetFile) {
        const key = canon(datasetFile.replace(".json", ""));
        return DATASET_SUMMARY_TEXT[key] || "当前展示该应用领域样例，可用于观察匿名节点在高阶关系中的聚类结构。";
    }

    function updateHeroInfo() {
        const scenario = getCurrentScenario();
        stageLabel.textContent = scenario.title;
        sampleDatasetLabel.textContent = state.currentDatasetName || "--";
        nodeCount.textContent = `${state.stats.shownNodeCount || "--"}/${state.stats.nodeCount || "--"}`;
        clusterCount.textContent = String(state.stats.clusterCount || "--");
        hyperedgeCount.textContent = String(state.stats.hyperedgeCount || "--");
        heroAcc.textContent = Number.isFinite(state.metrics.acc) ? state.metrics.acc.toFixed(3) : "--";
        heroNmi.textContent = Number.isFinite(state.metrics.nmi) ? state.metrics.nmi.toFixed(3) : "--";
        datasetSummaryCard.textContent = formatScenarioSummary(state.currentDatasetFile || "");
        if (datasetSummary) {
            datasetSummary.textContent = `当前示例：${state.currentDatasetName || "--"}。本步用于确定节点、属性和超边的业务含义，后续推荐、边界风险和局部关系解释都基于这个样例生成。`;
        }
    }

    function updateSelectionPanel() {
        const node = state.nodesById.get(state.selectedNodeId);
        if (!node) {
            selectionPanel.hidden = true;
            selectionPanel.innerHTML = "<h3>节点聚类解释</h3><p>搜索或点击任意节点后，这里将展示该节点的预测簇、同簇相似节点、共享超边邻居和边界风险。共享超边表示多个节点共同参与同一组高阶关系，是解释聚类归属和相关节点推荐的重要结构证据。</p>";
            return;
        }

        selectionPanel.hidden = false;
        const similarCount = node.top_similar.length;
        const neighborCount = node.hyperedge_neighbors.length;
        const incidentCount = node.edgeIds.length;
        const crossRatio = (node.crossClusterNeighborRatio * 100).toFixed(1);
        selectionPanel.innerHTML = `
            <h3>节点聚类解释</h3>
            <div class="node-detail-card">
                <div class="node-title">${state.currentDatasetName || "示例"} Node #${String(node.id).padStart(4, "0")}</div>
                <div class="node-meta">
                    <span>预测簇：Cluster ${node.cluster}</span>
                    <span>节点类型：匿名样本节点</span>
                    <span>分配强度：${(node.confidence || 0).toFixed(2)}</span>
                </div>
                <p class="node-explain">该节点被模型分配至 Cluster ${node.cluster}。解释依据包括两类：一是嵌入空间中与其接近的同簇节点，二是与其共享超边的结构邻居。</p>
                <p class="node-explain">同簇相似节点：${similarCount} 个；共享超边邻居：${neighborCount} 个；参与超边：${incidentCount} 条；跨簇邻居比例：${crossRatio}%。跨簇比例越高，说明该节点同时连接多个预测簇的可能性越强。</p>
                ${node.boundaryScore >= 0.45 ? `<div class="node-risk-warning">该节点存在较明显的跨簇连接，可能处于簇边界区域。建议结合共享超边进一步判断这些跨簇连接来自哪些高阶关系。</div>` : ""}
            </div>
        `;
    }

    function recommendItemHTML(item, type) {
        const scoreText = type === "similar" ? `相似度 ${Number(item.score || 0).toFixed(2)}` : `共享超边 ${item.shared_hyperedges || 0}`;
        let reason =
            item.reason ||
            (type === "similar"
                ? "同簇且嵌入距离较近，说明模型认为二者属性和局部结构相似。"
                : "与当前节点共享高阶超边，说明二者共同出现在同一组关系中，可作为结构相关性的依据。");
        if (type === "neighbor" && reason.includes("共享") && reason.includes("超边") && !reason.includes("说明")) {
            reason = `${reason} 这表示二者处在同一组高阶关系中，可帮助解释结构关联。`;
        }
        if (type === "similar" && reason.includes("同簇") && !reason.includes("说明")) {
            reason = `${reason} 这表示模型在嵌入空间中认为二者更接近。`;
        }
        return `<button type="button" class="recommend-item" data-node-id="${item.id}">
            <div class="recommend-main"><strong>Node ${item.id}</strong><span class="score">${scoreText}</span></div>
            <div class="recommend-tags"><span>Cluster ${item.cluster}</span><span>${type === "similar" ? "同簇相似" : "结构邻居"}</span></div>
            <p>${reason}</p>
        </button>`;
    }

    function updateRecommendationPanels() {
        const node = state.nodesById.get(state.selectedNodeId);
        if (!node) {
            recommendationPanel.innerHTML = "<p>请选择一个节点后查看相关节点证据。</p>";
            recommendReasonPanel.innerHTML = "<p class='panel-help'>请选择一个节点后查看解释依据。页面将分别说明预测簇一致性、共享超边和嵌入相似性提供的证据。</p>";
            return;
        }

        recommendationPanel.innerHTML = `
            <section class="recommend-section">
                <h4>嵌入相似节点</h4>
                ${node.top_similar.length ? node.top_similar.slice(0, 6).map((item) => recommendItemHTML(item, "similar")).join("") : "<p class='panel-help'>暂无同簇相似节点数据。</p>"}
            </section>
            <section class="recommend-section">
                <h4>共享超边邻居</h4>
                ${node.hyperedge_neighbors.length ? node.hyperedge_neighbors.slice(0, 6).map((item) => recommendItemHTML(item, "neighbor")).join("") : "<p class='panel-help'>暂无共享超边邻居数据。</p>"}
            </section>
        `;

        const formula = `score = 0.45 × 同簇匹配 + 0.35 × 共享超边得分 + 0.20 × 嵌入相似度`;
        recommendReasonPanel.innerHTML = `
            <div>
                <strong>同簇关系</strong>
                <span>如果两个节点落在同一预测簇，说明模型在无监督条件下认为它们更可能属于同一潜在主题、群体或类别。</span>
            </div>
            <div>
                <strong>共享超边</strong>
                <span>如果两个节点共享超边，说明它们共同参与同一组多节点关系。该信号可解释相关性来源，也可用于识别跨簇或混合关系。</span>
            </div>
            <div>
                <strong>嵌入相似度</strong>
                <span>嵌入距离越近，表示节点属性和局部结构综合后的表示越相似。当前权重：${formula}。</span>
            </div>
            <p class="panel-help">依据大赛规则，页面不使用真实标签；推荐与解释只依据预测簇、共享超边和嵌入相似度。</p>
        `;
    }

    function updateMetricPanels() {
        const m = state.metrics || {};
        const core = [
            ["ACC", m.acc, "公开结果中的准确率参考"],
            ["NMI", m.nmi, "公开结果中的归一化互信息参考"],
            ["ARI", m.ari, "公开结果中的随机一致性参考"],
            ["F1", m.f1, "公开结果中的综合参考指标"],
        ];
        metricPanel.innerHTML = core
            .filter((item) => Number.isFinite(item[1]))
            .map((item) => `<div class="metric-item"><strong>${item[1].toFixed(4)}</strong><span>${item[0]}：${item[2]}</span></div>`)
            .join("");
        const lowRatio = state.hyperedges.length ? state.hyperedges.filter((e) => e.purity < 0.55).length / state.hyperedges.length : 0;
        const boundaryCount = state.nodes.filter((n) => n.boundaryScore >= 0.45).length;
        const extra = [
            ["节点数", state.stats.shownNodeCount || state.nodes.length, "当前参与可视化的匿名对象规模"],
            ["预测簇数", state.stats.clusterCount || state.clusterStats.length, "模型自动划分出的潜在群体数量"],
            ["超边数", state.stats.hyperedgeCount || state.hyperedges.length, "多节点共同关系数量，用于解释群体连接"],
            ["低纯度超边比例", `${(lowRatio * 100).toFixed(1)}%`, "比例越高，说明跨簇混合关系越多"],
            ["边界节点数量", boundaryCount, "可能连接多个簇、需要重点解释的节点数"],
        ];
        hyperedgePanel.innerHTML = extra.map((item) => `<div class="metric-item"><strong>${item[1]}</strong><span>${item[0]}：${item[2]}</span></div>`).join("");
    }

    function updateClusterMatrix() {
        if (!state.clusterStats.length) {
            clusterMatrix.innerHTML = "<p class='panel-help'>当前样例暂无预测簇结构概览。</p>";
            return;
        }
        clusterMatrix.innerHTML = state.clusterStats
            .map((cluster) => {
                const boundaryCount = state.nodes.filter((node) => node.cluster === cluster.clusterId && node.boundaryScore >= 0.45).length;
                const cells = [
                    `节点:${cluster.size}`,
                    `平均强度:${cluster.avgConfidence.toFixed(2)}`,
                    `一致性:${cluster.consistency.toFixed(2)}`,
                    `边界节点:${boundaryCount}`,
                ]
                    .map((text) => `<div class="matrix-cell">${text}</div>`)
                    .join("");
                return `<div class="matrix-row"><div class="matrix-header"><strong>预测簇 C${cluster.clusterId}</strong><em>不使用真实标签</em></div><div class="matrix-cells">${cells}</div></div>`;
            })
            .join("");
    }

    function edgeItemHTML(edge, type = "risk") {
        const metaColor = type === "good" ? "good" : "risk";
        const text =
            type === "cross"
                ? "该超边连接多个预测簇，说明这组高阶关系横跨不同群体，是边界分析的重要线索。"
                : type === "max"
                    ? "该超边规模较大，代表许多节点共同参与同一组关系，可用于观察高阶关系密集区域。"
                    : "该超边纯度较低，说明同一组关系内混入了多个预测簇，属于模型较难判别的结构。";
        return `<button type="button" class="list-item ${metaColor}" data-edge-id="${edge.id}">
            <div class="list-title">Hyperedge ${edge.id}</div>
            <div class="list-meta">
                <span>节点 ${edge.size}</span>
                <span>簇数 ${edge.clusters.length}</span>
                <span>纯度 ${edge.purity.toFixed(3)}</span>
            </div>
            <p>${text}</p>
        </button>`;
    }

    function updateGlobalLists() {
        const topRisk = [...state.hyperedges]
            .filter((e) => e.mixed || e.purity < 0.7)
            .sort((a, b) => a.purity - b.purity || b.size - a.size)
            .slice(0, 10);
        topEdgeList.innerHTML = topRisk.length ? topRisk.map((e) => edgeItemHTML(e)).join("") : "<p class='panel-help'>暂无相关超边。</p>";

        const cross = [...state.hyperedges].filter((e) => e.mixed).sort((a, b) => a.purity - b.purity).slice(0, 10);
        crossEdgeList.innerHTML = cross.length ? cross.map((e) => edgeItemHTML(e, "cross")).join("") : "<p class='panel-help'>暂无跨簇超边。</p>";

        const boundaryNodes = [...state.nodes].sort((a, b) => b.boundaryScore - a.boundaryScore).slice(0, 12);
        boundaryNodeList.innerHTML = boundaryNodes.length
            ? boundaryNodes
                  .map((n) => `<button type="button" class="list-item risk" data-node-id="${n.id}">
                    <div class="list-title">Node ${n.id}</div>
                    <div class="list-meta">
                        <span>Cluster ${n.cluster}</span>
                        <span>边界分 ${n.boundaryScore.toFixed(2)}</span>
                        <span>跨簇比 ${(n.crossClusterNeighborRatio * 100).toFixed(1)}%</span>
                    </div>
                    <p>该节点的共享超边邻居分散在多个簇中，可能位于跨簇关系边界区域。</p>
                </button>`)
                  .join("")
            : "<p class='panel-help'>暂无边界节点。</p>";

        const maxEdges = [...state.hyperedges].sort((a, b) => b.size - a.size || a.purity - b.purity).slice(0, 10);
        maxHyperedgeList.innerHTML = maxEdges.length ? maxEdges.map((e) => edgeItemHTML(e, "max")).join("") : "<p class='panel-help'>暂无相关超边。</p>";

        highConfidenceClusterList.innerHTML = state.clusterStats
            .slice(0, 10)
            .map((c) => `<div class="list-item good">
                <div class="list-title">Cluster ${c.clusterId}</div>
                <div class="list-meta">
                    <span>节点 ${c.size}</span>
                    <span>平均强度 ${c.avgConfidence.toFixed(2)}</span>
                    <span>一致性 ${c.consistency.toFixed(2)}</span>
                </div>
                <p>该簇结构较稳定，簇内节点在属性表示和高阶关系上更一致。</p>
            </div>`)
            .join("");
    }

    function updatePanels() {
        updateHeroInfo();
        updateSelectionPanel();
        updateRecommendationPanels();
        updateMetricPanels();
        updateClusterMatrix();
        updateGlobalLists();
    }

    function screenFromWorld(world) {
        const w = clusterCanvas.clientWidth;
        const h = clusterCanvas.clientHeight;
        const drawableH = Math.max(1, h - CANVAS_CAPTION_HEIGHT);
        const pad = 52;
        const base = {
            x: pad + world.x * Math.max(1, w - pad * 2),
            y: pad + (1 - world.y) * Math.max(1, drawableH - pad * 2),
        };
        return {
            x: base.x * state.viewTransform.scale + state.viewTransform.panX,
            y: base.y * state.viewTransform.scale + state.viewTransform.panY,
        };
    }

    function worldFromScreen(screen) {
        const w = clusterCanvas.clientWidth;
        const h = clusterCanvas.clientHeight;
        const drawableH = Math.max(1, h - CANVAS_CAPTION_HEIGHT);
        const pad = 52;
        const scale = state.viewTransform.scale || 1;
        const baseX = (screen.x - state.viewTransform.panX) / scale;
        const baseY = (screen.y - state.viewTransform.panY) / scale;
        return {
            x: clamp((baseX - pad) / Math.max(1, w - pad * 2), 0.02, 0.98),
            y: clamp(1 - (baseY - pad) / Math.max(1, drawableH - pad * 2), 0.02, 0.98),
        };
    }

    function drawCanvasCaption(text = CANVAS_CAPTION_TEXT) {
        const w = clusterCanvas.clientWidth;
        const h = clusterCanvas.clientHeight;
        const top = Math.max(0, h - CANVAS_CAPTION_HEIGHT);

        ctx.save();
        ctx.fillStyle = "rgba(248, 250, 252, 0.92)";
        ctx.fillRect(0, top, w, CANVAS_CAPTION_HEIGHT);
        ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
        ctx.beginPath();
        ctx.moveTo(0, top + 0.5);
        ctx.lineTo(w, top + 0.5);
        ctx.stroke();
        ctx.fillStyle = "#475569";
        ctx.font = "700 12px Montserrat, Arial, sans-serif";
        ctx.textBaseline = "middle";
        ctx.fillText(text, 14, top + CANVAS_CAPTION_HEIGHT / 2);
        ctx.restore();
    }

    function resetViewTransform() {
        state.viewTransform.scale = 1;
        state.viewTransform.panX = 0;
        state.viewTransform.panY = 0;
    }

    function drawNode(point, node, options = {}) {
        const r = options.radius || 4;
        ctx.beginPath();
        ctx.arc(point.x, point.y, r, 0, Math.PI * 2);
        ctx.fillStyle = colorForNode(node);
        ctx.globalAlpha = options.alpha ?? 0.9;
        ctx.fill();
        ctx.globalAlpha = 1;
        if (options.stroke) {
            ctx.beginPath();
            ctx.arc(point.x, point.y, r + (options.strokePad || 2), 0, Math.PI * 2);
            ctx.strokeStyle = options.stroke;
            ctx.lineWidth = options.lineWidth || 1.6;
            if (options.dashed) ctx.setLineDash([4, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    function getLocalLayout(selectedNodeId) {
        if (state.localLayoutCache.has(selectedNodeId)) return state.localLayoutCache.get(selectedNodeId);
        const node = state.nodesById.get(selectedNodeId);
        if (!node) return null;

        const similarIds = new Set(node.top_similar.slice(0, 10).map((item) => Number(item.id)));
        const neighborIds = new Set(node.hyperedge_neighbors.slice(0, 12).map((item) => Number(item.id)));
        const incidentEdges = node.edgeIds.map((id) => state.hyperedgesById.get(id)).filter(Boolean).slice(0, 18);

        const allIds = new Set([node.id, ...similarIds, ...neighborIds]);
        incidentEdges.forEach((edge) => edge.memberIds.forEach((id) => allIds.add(id)));
        const members = [...allIds].map((id) => state.nodesById.get(id)).filter(Boolean);

        const roles = new Map();
        roles.set(node.id, "selected");
        similarIds.forEach((id) => roles.set(id, "similar"));
        neighborIds.forEach((id) => {
            if (!roles.has(id)) roles.set(id, "neighbor");
        });
        members.forEach((m) => {
            if (!roles.has(m.id)) roles.set(m.id, "other");
        });

        const points = new Map();
        points.set(node.id, { x: 0.5, y: 0.5 });
        placeRing(members.filter((m) => roles.get(m.id) === "similar"), points, 0.19);
        placeRing(members.filter((m) => roles.get(m.id) === "neighbor"), points, 0.29);
        placeRing(members.filter((m) => roles.get(m.id) === "other"), points, 0.39);

        const edges = incidentEdges
            .map((edge) => ({
                ...edge,
                memberIds: edge.memberIds.filter((id) => points.has(id)),
            }))
            .filter((edge) => edge.memberIds.length > 1);

        const layout = { points, roles, edges, selectedId: node.id };
        state.localLayoutCache.set(selectedNodeId, layout);
        return layout;
    }

    function placeRing(nodes, pointsMap, radius) {
        const total = Math.max(1, nodes.length);
        nodes
            .sort((a, b) => a.id - b.id)
            .forEach((node, idx) => {
                const theta = (Math.PI * 2 * idx) / total;
                pointsMap.set(node.id, {
                    x: clamp(0.5 + Math.cos(theta) * radius, 0.04, 0.96),
                    y: clamp(0.5 + Math.sin(theta) * radius, 0.04, 0.96),
                });
            });
    }

    function shouldDrawEdge(edge) {
        if (state.edgeFilter === "high") return edge.purity >= 0.8;
        if (state.edgeFilter === "medium") return edge.purity >= 0.55 && edge.purity < 0.8;
        if (state.edgeFilter === "low") return edge.purity < 0.55;
        if (state.edgeFilter === "mixed") return edge.mixed;
        return true;
    }

    function pickGlobalSampleNodes(limit = 260) {
        if (state.nodes.length <= limit) return [...state.nodes];
        const sampled = [];
        const step = Math.max(1, Math.floor(state.nodes.length / limit));
        for (let i = 0; i < state.nodes.length && sampled.length < limit; i += step) {
            sampled.push(state.nodes[i]);
        }
        return sampled;
    }

    function getGlobalOverviewLayout() {
        if (state.globalLayoutCache && state.globalLayoutCache.datasetFile === state.currentDatasetFile) {
            return state.globalLayoutCache.layout;
        }

        const sampledNodes = pickGlobalSampleNodes(260);
        const points = new Map(sampledNodes.map((node) => [node.id, { x: node.embed.x, y: node.embed.y }]));
        const sampledSet = new Set(sampledNodes.map((n) => n.id));
        const edges = [];
        for (const edge of state.hyperedges) {
            if (edges.length >= 340) break;
            if (!shouldDrawEdge(edge)) continue;
            const memberIds = edge.memberIds.filter((id) => sampledSet.has(id));
            if (memberIds.length < 2) continue;
            edges.push({ ...edge, memberIds });
        }

        const layout = { points, edges, selectedId: null, mode: "global" };
        state.globalLayoutCache = { datasetFile: state.currentDatasetFile, layout };
        return layout;
    }

    function drawGlobalView() {
        const layout = getGlobalOverviewLayout();
        if (!layout || !layout.points.size) {
            showEmpty("当前视图暂无可显示数据。");
            state.hitPoints = [];
            state.activeLayoutRef = null;
            return;
        }
        hideEmpty();
        state.activeLayoutRef = layout;
        state.hitPoints = [];

        const screenPoints = new Map();
        layout.points.forEach((world, id) => screenPoints.set(id, screenFromWorld(world)));

        if (state.showHyperedges) {
            layout.edges.forEach((edge) => {
                if (!shouldDrawEdge(edge)) return;
                const members = edge.memberIds.map((id) => screenPoints.get(id)).filter(Boolean);
                if (members.length < 2) return;
                const cx = members.reduce((sum, p) => sum + p.x, 0) / members.length;
                const cy = members.reduce((sum, p) => sum + p.y, 0) / members.length;
                const color = edgeColor(edge);
                members.forEach((p) => {
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.quadraticCurveTo((cx + p.x) / 2, (cy + p.y) / 2 - 8, p.x, p.y);
                    ctx.strokeStyle = color;
                    ctx.globalAlpha = 0.16;
                    ctx.lineWidth = 0.9;
                    ctx.stroke();
                    ctx.globalAlpha = 1;
                });
                ctx.beginPath();
                ctx.arc(cx, cy, Math.max(6, Math.min(12, members.length * 0.5)), 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.14;
                ctx.fill();
                ctx.globalAlpha = 1;
                state.hitPoints.push({ type: "edge", id: edge.id, x: cx, y: cy, r: 10 });
            });
        }

        layout.points.forEach((world, id) => {
            const node = state.nodesById.get(id);
            const point = screenPoints.get(id);
            const isHovered = state.hoveredNodeId === id;
            drawNode(point, node, { radius: 3.3, stroke: isHovered ? "#0f172a" : null, alpha: 0.8 });
            state.hitPoints.push({ type: "node", id, x: point.x, y: point.y, r: 7 });
        });

        drawCanvasCaption();
    }

    function drawHypergraphView() {
        const selected = state.nodesById.get(state.selectedNodeId);
        if (!selected) {
            showEmpty("请选择一个节点后查看局部超图关系。");
            state.hitPoints = [];
            state.activeLayoutRef = null;
            return;
        }
        const layout = getLocalLayout(selected.id);
        if (!layout) {
            showEmpty("当前视图暂无可显示数据。");
            state.hitPoints = [];
            state.activeLayoutRef = null;
            return;
        }
        hideEmpty();
        state.activeLayoutRef = layout;
        state.hitPoints = [];

        const screenPoints = new Map();
        layout.points.forEach((world, id) => screenPoints.set(id, screenFromWorld(world)));

        if (state.showHyperedges) {
            layout.edges.forEach((edge) => {
                if (!shouldDrawEdge(edge)) return;
                const members = edge.memberIds.map((id) => screenPoints.get(id)).filter(Boolean);
                if (members.length < 2) return;
                const cx = members.reduce((sum, p) => sum + p.x, 0) / members.length;
                const cy = members.reduce((sum, p) => sum + p.y, 0) / members.length;
                const color = edgeColor(edge);
                members.forEach((p) => {
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.quadraticCurveTo((cx + p.x) / 2, (cy + p.y) / 2 - 10, p.x, p.y);
                    ctx.strokeStyle = color;
                    ctx.globalAlpha = 0.22;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.globalAlpha = 1;
                });
                ctx.beginPath();
                ctx.arc(cx, cy, Math.max(7, Math.min(14, members.length * 0.58)), 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.14;
                ctx.fill();
                ctx.globalAlpha = 1;
                state.hitPoints.push({ type: "edge", id: edge.id, x: cx, y: cy, r: 11 });
            });
        }

        const similarSet = new Set((selected.top_similar || []).map((item) => Number(item.id)));
        const neighborSet = new Set((selected.hyperedge_neighbors || []).map((item) => Number(item.id)));
        layout.points.forEach((world, id) => {
            const node = state.nodesById.get(id);
            const point = screenFromWorld(world);
            const isSelected = id === selected.id;
            const isSimilar = similarSet.has(id);
            const isNeighbor = neighborSet.has(id);
            const isHovered = state.hoveredNodeId === id;

            let stroke = null;
            let radius = isSelected ? 6 : 4;
            if (isSelected) stroke = "#0f172a";
            else if (state.quickMode === "node-similar" && isSimilar) stroke = "#1d4ed8";
            else if (state.quickMode === "node-hyperedge-neighbors" && isNeighbor) stroke = "#d97706";
            else if (state.quickMode === "node-boundary-risk" && node.boundaryScore >= 0.45) stroke = "#dc2626";
            else if (isHovered) stroke = "#0f172a";

            drawNode(point, node, { radius, stroke, dashed: stroke === "#d97706", alpha: selected ? 0.9 : 0.78 });
            state.hitPoints.push({ type: "node", id, x: point.x, y: point.y, r: radius + 5 });
        });
    }

    function drawBoundaryView() {
        if (!state.nodes.length) {
            showEmpty("当前视图暂无可显示数据。");
            state.hitPoints = [];
            state.activeLayoutRef = null;
            return;
        }
        hideEmpty();
        state.activeLayoutRef = null;
        const selected = state.nodesById.get(state.selectedNodeId);
        const topRiskEdges = state.hyperedges.filter((e) => e.purity < 0.55 || e.mixed).sort((a, b) => a.purity - b.purity).slice(0, 50);
        if (state.showHyperedges) {
            topRiskEdges.forEach((edge) => {
                if (!shouldDrawEdge(edge)) return;
                const members = edge.members.map((n) => screenFromWorld(n.embed));
                if (members.length < 2) return;
                const color = edge.mixed ? "#ea580c" : "#ef4444";
                for (let i = 0; i < Math.min(10, members.length); i += 1) {
                    const a = members[i];
                    const b = members[(i + 1) % members.length];
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = color;
                    ctx.globalAlpha = 0.12;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.globalAlpha = 1;
                }
            });
        }

        state.hitPoints = [];
        state.nodes.forEach((node) => {
            const point = screenFromWorld(node.embed);
            const risk = node.boundaryScore;
            const isSelected = selected && node.id === selected.id;
            const radius = isSelected ? 6 : 3.8;
            drawNode(point, node, {
                radius,
                stroke: isSelected ? "#0f172a" : risk >= 0.45 ? (risk >= 0.65 ? "#dc2626" : "#f97316") : null,
                dashed: !isSelected && risk >= 0.45,
                lineWidth: isSelected ? 2 : 1.2,
            });
            state.hitPoints.push({ type: "node", id: node.id, x: point.x, y: point.y, r: radius + 4 });
        });
    }

    function render() {
        clearCanvas();
        if (state.loading) return;
        if (!state.nodes.length) {
            showEmpty("当前视图暂无可显示数据。");
            state.activeLayoutRef = null;
            return;
        }
        if (state.currentView === "global") {
            drawGlobalView();
            return;
        }
        if (state.currentView === "boundary") {
            drawBoundaryView();
            return;
        }
        drawHypergraphView();
    }

    function findHitAt(point) {
        let best = null;
        let bestDist = 1e9;
        state.hitPoints.forEach((item) => {
            const d = Math.hypot(point.x - item.x, point.y - item.y);
            if (d <= item.r && d < bestDist) {
                bestDist = d;
                best = item;
            }
        });
        return best;
    }

    function dragEdgeByScreenDelta(edgeId, currentScreen, prevScreen) {
        if (!state.activeLayoutRef?.points || !Array.isArray(state.activeLayoutRef.edges)) return;
        const edge = state.activeLayoutRef.edges.find((e) => e.id === edgeId);
        if (!edge) return;
        const worldNow = worldFromScreen(currentScreen);
        const worldPrev = worldFromScreen(prevScreen);
        const dx = worldNow.x - worldPrev.x;
        const dy = worldNow.y - worldPrev.y;
        edge.memberIds.forEach((id) => {
            if (!state.activeLayoutRef.points.has(id)) return;
            const p = state.activeLayoutRef.points.get(id);
            state.activeLayoutRef.points.set(id, {
                x: clamp(p.x + dx, 0.02, 0.98),
                y: clamp(p.y + dy, 0.02, 0.98),
            });
        });
    }

    function showTooltip(nodeId, point) {
        const node = state.nodesById.get(nodeId);
        if (!node || !clusterTooltip) return;
        clusterTooltip.innerHTML = `<strong>Node ${node.id}</strong><br>预测簇：C${node.cluster}<br>分配强度：${(node.confidence || 0).toFixed(2)}<br>边界分：${node.boundaryScore.toFixed(2)}`;
        clusterTooltip.style.left = `${Math.min(clusterCanvas.clientWidth - 250, point.x + 16)}px`;
        clusterTooltip.style.top = `${Math.max(10, point.y + 12)}px`;
        clusterTooltip.classList.add("show");
    }

    function hideTooltip() {
        clusterTooltip?.classList.remove("show");
    }

    function selectNode(nodeId) {
        const node = state.nodesById.get(Number(nodeId));
        if (!node) return false;
        state.selectedNodeId = node.id;
        state.currentView = "hypergraph";
        [analysisTabs, stepTabs].forEach((container) => {
            container?.querySelectorAll("[data-view]").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === "hypergraph"));
        });
        updateSelectionPanel();
        updateRecommendationPanels();
        setNarrative(`已选中节点 ${node.id}。右侧将从预测簇归属、同簇相似节点、共享超边邻居和边界风险四个方面解释该节点。`);
        render();
        return true;
    }

    function applyQuickAction(action) {
        const node = state.nodesById.get(state.selectedNodeId);
        if (!node) {
            setStatus("请先搜索或点击一个节点，再查看局部解释。");
            return;
        }

        state.quickMode = action;
        if (action === "node-incident-edges") {
            switchView("hypergraph");
            setNarrative(`正在查看节点 ${node.id} 的局部高阶关系。线条表示多个节点共同出现、共同参与或共同关联的超边。`);
        } else if (action === "node-boundary-risk") {
            switchView("boundary");
            setNarrative(`正在查看节点 ${node.id} 的边界风险。边界风险越高，说明该节点越可能同时连接多个预测簇。`);
        } else if (action === "focus-selected-node") {
            switchView("hypergraph");
            setNarrative(`已聚焦节点 ${node.id}。中心节点周围展示同簇相似节点和共享超边邻居，用于解释其聚类归属。`);
        } else if (action === "node-similar") {
            switchView("hypergraph");
            setNarrative(`正在查看节点 ${node.id} 的同簇相似节点。这类节点用于说明模型表示空间中的相似性证据。`);
        } else if (action === "node-hyperedge-neighbors") {
            switchView("hypergraph");
            setNarrative(`正在查看节点 ${node.id} 的共享超边邻居。这类节点用于说明多节点高阶关系中的结构证据。`);
        }
        render();
    }

    function switchView(viewName) {
        if (!["global", "hypergraph", "boundary"].includes(viewName)) return;
        state.currentView = viewName;
        [analysisTabs, stepTabs].forEach((container) => {
            container?.querySelectorAll("[data-view]").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === viewName));
        });

        if (viewName === "global") {
            setNarrative("当前为整体聚类视图。不同颜色表示不同预测簇，可用于观察模型发现的潜在群体结构。");
        } else if (viewName === "hypergraph") {
            const hasSelected = Boolean(state.selectedNodeId);
            setNarrative(
                hasSelected
                    ? "当前为局部高阶关系视图。中心节点周围展示同簇相似节点和共享超边邻居。"
                    : "请先点击或搜索节点，随后查看该节点的局部高阶关系。"
            );
        } else {
            setNarrative("当前为边界风险视图，用于观察哪些节点或超边可能连接多个预测簇。");
        }
        render();
    }
    function updateScenarioAndDatasetUI() {
        updateScenarioSelects();
        renderScenarioCards();
    }

    async function loadDataset(file) {
        if (!file) return;
        const token = ++state.loadToken;
        setLoading(true, "数据加载中...");
        setStatus("数据加载中...");
        try {
            await nextPaint();
            if (token !== state.loadToken) return;

            let raw = null;
            if (window.CLUSTER_DATASETS && window.CLUSTER_DATASETS[file]) raw = window.CLUSTER_DATASETS[file];
            else {
                const response = await fetch(`data/cluster/${file}`, { cache: "no-store" });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                raw = await response.json();
            }

            if (token !== state.loadToken) return;
            const parsed = preprocessDataset(raw);
            if (token !== state.loadToken) return;
            state.currentDatasetFile = file;
            state.currentDatasetName = getDisplayDatasetLabel(file);
            state.nodes = parsed.nodes;
            state.nodesById = parsed.nodesById;
            state.hyperedges = parsed.hyperedges;
            state.hyperedgesById = parsed.hyperedgesById;
            state.metrics = parsed.metrics;
            state.stats = parsed.stats;
            state.clusterStats = parsed.clusterStats;
            state.selectedNodeId = null;
            state.hoveredNodeId = null;
            state.quickMode = null;
            state.localLayoutCache.clear();
            state.globalLayoutCache = null;
            resetViewTransform();

            updatePanels();
            setNarrative("数据已加载。颜色表示模型预测簇；点击任意节点后，右侧将展示该节点的局部高阶关系解释。");
            setStatus(`已加载 ${parsed.dataset}。可点击图中节点，或输入节点编号进行解释。`);
            render();
        } catch (error) {
            state.nodes = [];
            state.hyperedges = [];
            state.nodesById = new Map();
            state.hyperedgesById = new Map();
            showEmpty("数据加载失败，请检查 data/cluster/cluster-data.js 是否正确加载。");
            setStatus("数据加载失败，请检查 data/cluster/cluster-data.js 是否正确加载。");
        } finally {
            if (token === state.loadToken) setLoading(false);
        }
    }

    function setScenario(scenarioId, preferredLabel = null) {
        state.currentScenarioId = scenarioId;
        const scenario = getCurrentScenario();
        const entries = getAvailableScenarioDatasetEntries(scenario);
        if (!entries.length) {
            updateScenarioAndDatasetUI();
            setStatus(`应用领域 ${scenario.title} 暂无可用示例数据。`);
            return;
        }
        let entry = preferredLabel ? entries.find((item) => item.label === preferredLabel)?.entry : null;
        if (!entry) {
            const preferred = entries.find((item) => canon(item.label) === canon(scenario.defaultDataset));
            entry = preferred?.entry || entries[0].entry;
        }
        state.currentDatasetFile = entry.file;
        state.currentDatasetName = getDisplayDatasetLabel(entry.file);
        updateScenarioAndDatasetUI();
        if (datasetSelect) datasetSelect.value = entry.file;
        loadDataset(entry.file);
    }

    function bindManifest(manifest) {
        state.manifest = manifest;
        state.availableDatasetMap.clear();
        (manifest.datasets || []).forEach((item) => {
            const key1 = canon(item.name);
            const key2 = canon(item.file.replace(".json", ""));
            state.availableDatasetMap.set(key1, item);
            state.availableDatasetMap.set(key2, item);
        });

        if (scenarioSelect) {
            scenarioSelect.innerHTML = APPLICATION_SCENARIOS.map((s) => `<option value="${s.id}">${s.title}</option>`).join("");
            scenarioSelect.value = "academic";
        }
        state.currentScenarioId = "academic";
        updateScenarioAndDatasetUI();
        setScenario("academic", "Cora");
    }

    async function loadManifest() {
        try {
            if (window.CLUSTER_DATA_MANIFEST) {
                bindManifest(window.CLUSTER_DATA_MANIFEST);
                return;
            }
            const response = await fetch("data/cluster/manifest.json", { cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            bindManifest(await response.json());
        } catch (error) {
            showEmpty("数据加载失败，请检查 data/cluster/cluster-data.js 是否正确加载。");
            setStatus("数据加载失败，请检查 data/cluster/cluster-data.js 是否正确加载。");
        }
    }

    function handleSearch() {
        const value = Number(searchInput.value);
        if (!Number.isFinite(value)) {
            const max = Math.max(0, (state.stats.shownNodeCount || state.nodes.length) - 1);
            setStatus(`未找到该节点，请输入 0 到 ${max} 范围内的节点编号。`);
            return;
        }
        if (!selectNode(value)) {
            const max = Math.max(0, (state.stats.shownNodeCount || state.nodes.length) - 1);
            setStatus(`未找到该节点，请输入 0 到 ${max} 范围内的节点编号。`);
            return;
        }
        setStatus(`已定位节点 ${value}。`);
    }

    function bindEvents() {
        startButton?.addEventListener("click", () => analysisMain?.scrollIntoView({ behavior: "smooth", block: "start" }));
        viewScenarioBtn?.addEventListener("click", () => scenarioSection?.scrollIntoView({ behavior: "smooth", block: "start" }));

        scenarioGrid?.addEventListener("click", (event) => {
            const datasetBtn = event.target.closest("[data-dataset-label]");
            if (datasetBtn) {
                setScenario(datasetBtn.dataset.scenarioId, datasetBtn.dataset.datasetLabel);
                return;
            }
            const enterBtn = event.target.closest("[data-enter-scenario]");
            if (enterBtn) {
                setScenario(enterBtn.dataset.enterScenario, enterBtn.dataset.enterDatasetLabel || null);
                analysisMain?.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        });

        scenarioSelect?.addEventListener("change", () => setScenario(scenarioSelect.value));
        datasetSelect?.addEventListener("change", () => {
            const file = datasetSelect.value;
            const scenario = getCurrentScenario();
            const entries = getAvailableScenarioDatasetEntries(scenario);
            const matched = entries.find((item) => item.entry.file === file);
            if (matched) loadDataset(file);
        });

        analysisTabs?.addEventListener("click", (event) => {
            const btn = event.target.closest("[data-view]");
            if (btn) switchView(btn.dataset.view);
        });
        stepTabs?.addEventListener("click", (event) => {
            const btn = event.target.closest("[data-view]");
            if (btn) switchView(btn.dataset.view);
        });

        colorMode?.addEventListener("change", () => {
            state.colorMode = colorMode.value;
            updateSelectionPanel();
            const colorText =
                state.colorMode === "purity"
                        ? "风险强度着色用于突出边界节点和低纯度关系区域。"
                        : "预测簇着色用于观察模型自动划分出的群体结构。";
            setNarrative(`Step 2 已切换着色模式。${colorText}`);
            render();
        });
        edgeFilter?.addEventListener("change", () => {
            state.edgeFilter = edgeFilter.value;
            const filterText =
                {
                    all: "显示全部超边，用于完整查看高阶关系。",
                    high: "只显示高纯度超边，用于观察簇内较稳定的群体关系。",
                    medium: "只显示中等纯度超边，用于观察可能含有边界样本的关系。",
                    low: "只显示低纯度超边，用于定位模型难判别的混合关系。",
                    mixed: "只显示跨簇超边，用于解释不同预测簇之间的连接来源。",
                }[state.edgeFilter] || "已更新超边筛选。";
            setNarrative(`Step 2 已应用超边筛选。${filterText}`);
            render();
        });
        showHyperedges?.addEventListener("change", () => {
            state.showHyperedges = showHyperedges.checked;
            syncHyperedgeExplain();
            setNarrative(
                state.showHyperedges
                    ? "Step 2 已显示超边链接。超边用于说明多个节点共同参与的高阶关系。"
                    : "Step 2 已隐藏超边链接。当前更适合观察节点分布，但会减少结构解释信息。"
            );
            render();
        });

        searchButton?.addEventListener("click", handleSearch);
        searchInput?.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                handleSearch();
            }
        });

        relayoutButton?.addEventListener("click", () => {
            if (state.selectedNodeId) state.localLayoutCache.delete(state.selectedNodeId);
            else state.globalLayoutCache = null;
            setStatus("已重排当前视图布局。");
            render();
        });
        resetButton?.addEventListener("click", () => {
            state.selectedNodeId = null;
            state.hoveredNodeId = null;
            state.quickMode = null;
            resetViewTransform();
            hideTooltip();
            updateSelectionPanel();
            updateRecommendationPanels();
            setStatus("已重置当前选择。");
            setNarrative("已重置当前选择。当前回到全局分析视图，可重新选择节点查看局部超图关系解释。");
            render();
        });
        exportButton?.addEventListener("click", () => {
            const link = document.createElement("a");
            link.download = `intrahc-${state.currentDatasetName || "sample"}-${state.currentView}.png`;
            link.href = clusterCanvas.toDataURL("image/png");
            link.click();
        });

        recommendationPanel?.addEventListener("click", (event) => {
            const btn = event.target.closest("[data-node-id]");
            if (!btn) return;
            selectNode(Number(btn.dataset.nodeId));
        });
        boundaryNodeList?.addEventListener("click", (event) => {
            const btn = event.target.closest("[data-node-id]");
            if (!btn) return;
            switchView("boundary");
            selectNode(Number(btn.dataset.nodeId));
        });
        [topEdgeList, crossEdgeList, maxHyperedgeList].forEach((container) => {
            container?.addEventListener("click", (event) => {
                const btn = event.target.closest("[data-edge-id]");
                if (!btn) return;
                const edge = state.hyperedgesById.get(Number(btn.dataset.edgeId));
                if (!edge || !edge.memberIds.length) return;
                switchView("hypergraph");
                selectNode(edge.memberIds[0]);
            });
        });

        document.querySelector(".analysis-chips")?.addEventListener("click", (event) => {
            const btn = event.target.closest("button[data-action]");
            if (!btn) return;
            applyQuickAction(btn.dataset.action);
        });

        clusterCanvas.addEventListener("pointerdown", (event) => {
            const point = getCanvasPoint(event);
            const hit = findHitAt(point);

            state.pointer.mode = "pan";
            state.pointer.nodeId = null;
            state.pointer.edgeId = null;

            if (hit && (state.currentView === "global" || state.currentView === "hypergraph")) {
                if (hit.type === "node") {
                    state.pointer.mode = "drag-node";
                    state.pointer.nodeId = hit.id;
                } else if (hit.type === "edge") {
                    state.pointer.mode = "drag-edge";
                    state.pointer.edgeId = hit.id;
                }
            }

            state.pointer.moved = false;
            state.pointer.last = point;
            clusterCanvas.setPointerCapture?.(event.pointerId);
        });

        clusterCanvas.addEventListener("pointermove", (event) => {
            const point = getCanvasPoint(event);

            if (state.pointer.mode === "drag-node" && state.pointer.nodeId !== null && state.pointer.last) {
                const moved = Math.hypot(point.x - state.pointer.last.x, point.y - state.pointer.last.y) > 1;
                state.pointer.moved = state.pointer.moved || moved;
                if (state.activeLayoutRef?.points?.has(state.pointer.nodeId)) {
                    state.activeLayoutRef.points.set(state.pointer.nodeId, worldFromScreen(point));
                    render();
                }
                state.pointer.last = point;
                hideTooltip();
                return;
            }

            if (state.pointer.mode === "drag-edge" && state.pointer.edgeId !== null && state.pointer.last) {
                const moved = Math.hypot(point.x - state.pointer.last.x, point.y - state.pointer.last.y) > 1;
                state.pointer.moved = state.pointer.moved || moved;
                dragEdgeByScreenDelta(state.pointer.edgeId, point, state.pointer.last);
                state.pointer.last = point;
                render();
                hideTooltip();
                return;
            }

            if (state.pointer.mode === "pan" && state.pointer.last) {
                const dx = point.x - state.pointer.last.x;
                const dy = point.y - state.pointer.last.y;
                state.pointer.moved = state.pointer.moved || Math.hypot(dx, dy) > 1;
                state.viewTransform.panX += dx;
                state.viewTransform.panY += dy;
                state.pointer.last = point;
                render();
                hideTooltip();
                return;
            }

            const hit = findHitAt(point);
            state.hoveredNodeId = hit?.type === "node" ? hit.id : null;
            if (hit?.type === "node") showTooltip(hit.id, point);
            else hideTooltip();
            render();
        });

        clusterCanvas.addEventListener("pointerup", (event) => {
            const point = getCanvasPoint(event);
            const hit = findHitAt(point);

            if (state.pointer.mode === "drag-node" && state.pointer.nodeId !== null && !state.pointer.moved) {
                selectNode(state.pointer.nodeId);
                setStatus(`已选中节点 ${state.pointer.nodeId}。`);
            } else if ((state.pointer.mode === "drag-edge" || state.pointer.mode === "pan") && !state.pointer.moved) {
                if (hit?.type === "node") {
                    selectNode(hit.id);
                    setStatus(`已选中节点 ${hit.id}。`);
                }
            }

            state.pointer.mode = null;
            state.pointer.nodeId = null;
            state.pointer.edgeId = null;
            state.pointer.moved = false;
            state.pointer.last = null;
            clusterCanvas.releasePointerCapture?.(event.pointerId);
        });

        clusterCanvas.addEventListener("pointercancel", () => {
            state.pointer.mode = null;
            state.pointer.nodeId = null;
            state.pointer.edgeId = null;
            state.pointer.moved = false;
            state.pointer.last = null;
        });

        clusterCanvas.addEventListener("wheel", (event) => {
            event.preventDefault();
            const point = getCanvasPoint(event);
            const factor = event.deltaY > 0 ? 0.92 : 1.08;
            const prevScale = state.viewTransform.scale;
            const nextScale = clamp(prevScale * factor, 0.55, 3.2);
            if (nextScale === prevScale) return;
            const anchorX = (point.x - state.viewTransform.panX) / prevScale;
            const anchorY = (point.y - state.viewTransform.panY) / prevScale;
            state.viewTransform.scale = nextScale;
            state.viewTransform.panX = point.x - anchorX * nextScale;
            state.viewTransform.panY = point.y - anchorY * nextScale;
            render();
            hideTooltip();
        }, { passive: false });

        clusterCanvas.addEventListener("mouseleave", () => {
            state.hoveredNodeId = null;
            hideTooltip();
            render();
        });

        window.addEventListener("resize", resizeCanvas);
    }

    function bootstrap() {
        bindEvents();
        if (showHyperedges) showHyperedges.checked = false;
        syncHyperedgeExplain();
        resizeCanvas();
        loadManifest();
    }

    bootstrap();
});
