window.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("clusterCanvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const datasetSelect = document.getElementById("datasetSelect");
    const metricPanel = document.getElementById("metricPanel");
    const hyperedgePanel = document.getElementById("hyperedgePanel");
    const selectionPanel = document.getElementById("selectionPanel");
    const clusterMatrix = document.getElementById("clusterMatrix");
    const topEdgeList = document.getElementById("topEdgeList");
    const recommendationPanel = document.getElementById("recommendationPanel");
    const analysisChips = document.querySelector(".analysis-chips");
    const dataStatus = document.getElementById("dataStatus");
    const relayoutButton = document.getElementById("relayoutView");
    const resetFocusButton = document.getElementById("resetFocus");
    const exportButton = document.getElementById("exportView");
    const surveyViewButton = document.getElementById("surveyView");
    const searchNodeButton = document.getElementById("searchNode");
    const nodeSearch = document.getElementById("nodeSearch");
    const colorMode = document.getElementById("colorMode");
    const edgeFilter = document.getElementById("edgeFilter");
    const showHyperedges = document.getElementById("showHyperedges");
    const tooltip = document.getElementById("clusterTooltip");
    const stageLabel = document.getElementById("stageLabel");
    const nodeCountLabel = document.getElementById("nodeCount");
    const clusterCountLabel = document.getElementById("clusterCount");
    const hyperedgeCountLabel = document.getElementById("hyperedgeCount");

    const clusterPalette = ["#246bfe", "#00a884", "#f59e0b", "#ef4444", "#7c3aed", "#0891b2", "#65a30d", "#db2777", "#475569", "#ea580c"];
    const labelPalette = ["#0f766e", "#b45309", "#6d28d9", "#be123c", "#0369a1", "#4d7c0f", "#a21caf", "#334155"];

    const state = {
        width: 0,
        height: 0,
        elapsed: 0,
        lastTime: 0,
        payload: null,
        nodes: [],
        clusters: [],
        hyperedges: [],
        hyperedgeById: new Map(),
        filteredEdges: [],
        filteredEdgeKey: null,
        selectionCache: null,
        selectionCacheKey: null,
        labelBoxes: [],
        hoveredNode: null,
        hoveredEdge: null,
        selected: null,
        drag: {
            type: null,
            node: null,
            edge: null,
            moved: false,
            pointerId: null,
            startX: 0,
            startY: 0,
            offsetX: 0,
            offsetY: 0,
            targetX: 0,
            targetY: 0,
            lastX: 0,
            lastY: 0,
            memberOffsets: [],
            settleUntil: 0,
        },
    };

    function invalidateDerivedState() {
        state.filteredEdges = [];
        state.filteredEdgeKey = null;
        state.selectionCache = null;
        state.selectionCacheKey = null;
    }

    function setStatus(message) {
        if (dataStatus) dataStatus.textContent = message;
    }

    function colorFor(value, mode) {
        const palette = mode === "label" ? labelPalette : clusterPalette;
        const numeric = Number.isFinite(Number(value)) ? Number(value) : 0;
        return palette[Math.abs(numeric) % palette.length];
    }

    function hexToRgba(hex, alpha) {
        const value = hex.replace("#", "");
        const r = parseInt(value.slice(0, 2), 16);
        const g = parseInt(value.slice(2, 4), 16);
        const b = parseInt(value.slice(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function roundedRect(x, y, width, height, radius) {
        const r = Math.min(radius, width / 2, height / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + width, y, x + width, y + height, r);
        ctx.arcTo(x + width, y + height, x, y + height, r);
        ctx.arcTo(x, y + height, x, y, r);
        ctx.arcTo(x, y, x + width, y, r);
        ctx.closePath();
    }

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        state.width = rect.width;
        state.height = rect.height;
        canvas.width = Math.floor(rect.width * ratio);
        canvas.height = Math.floor(rect.height * ratio);
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        rebuildScene({ keepPositions: true });
    }

    function scalePoint(node) {
        const paddingX = Math.max(52, state.width * 0.08);
        const paddingY = Math.max(58, state.height * 0.1);
        return {
            x: paddingX + node.x * Math.max(1, state.width - paddingX * 2),
            y: paddingY + (1 - node.y) * Math.max(1, state.height - paddingY * 2),
        };
    }

    function buildClusters(nodes) {
        const groups = new Map();
        nodes.forEach((node) => {
            if (!groups.has(node.cluster)) groups.set(node.cluster, []);
            groups.get(node.cluster).push(node);
        });

        return Array.from(groups.entries()).map(([clusterId, members]) => {
            const x = members.reduce((sum, node) => sum + node.tx, 0) / members.length;
            const y = members.reduce((sum, node) => sum + node.ty, 0) / members.length;
            const radius = Math.max(50, Math.min(150, Math.sqrt(members.length) * 10 + Math.min(state.width, state.height) * 0.03));
            return {
                id: Number(clusterId),
                label: `C${clusterId}`,
                color: colorFor(clusterId, "cluster"),
                x,
                y,
                radius,
                size: members.length,
                members,
                edgeIds: [],
                labelDistribution: distribution(members, "label"),
                avgHyperedgePurity: 0,
                mixedHyperedgeCount: 0,
            };
        });
    }

    function distribution(items, key) {
        const counts = new Map();
        items.forEach((item) => counts.set(item[key], (counts.get(item[key]) || 0) + 1));
        return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
    }

    function analyzeHyperedge(edge) {
        const members = edge.nodes.map((idx) => state.nodes[idx]).filter(Boolean);
        const clusterCounts = distribution(members, "cluster");
        const labelCounts = distribution(members, "label");
        const dominant = clusterCounts[0]?.[0] ?? null;
        const dominantCount = clusterCounts[0]?.[1] ?? 0;
        const purity = members.length ? dominantCount / members.length : 0;
        const cx = members.length ? members.reduce((sum, node) => sum + node.x, 0) / members.length : 0;
        const cy = members.length ? members.reduce((sum, node) => sum + node.y, 0) / members.length : 0;
        return {
            ...edge,
            members,
            clusterCounts,
            labelCounts,
            dominant,
            purity,
            mixed: clusterCounts.length > 1,
            x: cx,
            y: cy,
        };
    }

    function rebuildScene(options = {}) {
        if (!state.payload || !state.width || !state.height) return;
        const keepPositions = Boolean(options.keepPositions);
        invalidateDerivedState();

        state.nodes = state.payload.nodes.map((raw, index) => {
            const point = scalePoint(raw);
            const previous = keepPositions ? state.nodes.find((node) => node.id === raw.id) : null;
            return {
                id: raw.id,
                index,
                cluster: raw.cluster,
                label: raw.label,
                tx: point.x,
                ty: point.y,
                x: previous ? previous.x : point.x + (Math.random() - 0.5) * 120,
                y: previous ? previous.y : point.y + (Math.random() - 0.5) * 120,
                vx: 0,
                vy: 0,
                phase: previous ? previous.phase : Math.random() * Math.PI * 2,
                size: previous ? previous.size : 3.5 + Math.random() * 2,
                pinned: previous ? previous.pinned : false,
            };
        });

        state.clusters = buildClusters(state.nodes);
        state.nodes.forEach((node) => {
            node.edgeIds = [];
        });
        state.hyperedges = (state.payload.hyperedges || []).map(analyzeHyperedge).filter((edge) => edge.members.length > 1);
        state.hyperedgeById = new Map(state.hyperedges.map((edge) => [edge.id, edge]));
        const clusterById = new Map(state.clusters.map((cluster) => [cluster.id, cluster]));
        state.hyperedges.forEach((edge) => {
            edge.members.forEach((node) => node.edgeIds.push(edge.id));
            const touchedClusters = new Set(edge.members.map((node) => node.cluster));
            touchedClusters.forEach((clusterId) => {
                const cluster = clusterById.get(clusterId);
                if (cluster) cluster.edgeIds.push(edge.id);
            });
        });
        state.clusters.forEach((cluster) => {
            const relatedEdges = cluster.edgeIds.map((edgeId) => state.hyperedgeById.get(edgeId)).filter(Boolean);
            cluster.avgHyperedgePurity = relatedEdges.length
                ? relatedEdges.reduce((sum, edge) => sum + edge.purity, 0) / relatedEdges.length
                : 0;
            cluster.mixedHyperedgeCount = relatedEdges.filter((edge) => edge.mixed).length;
        });
        updateHyperedgeSummary();
        renderClusterMatrix();
        renderTopEdges();
        renderSelectionPanel();
        renderRecommendations();
    }

    function filteredHyperedges() {
        const filter = edgeFilter?.value || "all";
        if (state.filteredEdgeKey === filter) return state.filteredEdges;
        state.filteredEdgeKey = filter;
        state.filteredEdges = state.hyperedges.filter((edge) => {
            if (filter === "high") return edge.purity >= 0.8;
            if (filter === "medium") return edge.purity >= 0.55 && edge.purity < 0.8;
            if (filter === "low") return edge.purity < 0.55;
            if (filter === "mixed") return edge.mixed;
            return true;
        });
        return state.filteredEdges;
    }

    function relatedSelectionSets() {
        const key = state.selected ? `${state.selected.type}:${state.selected.id}` : "none";
        if (state.selectionCacheKey === key && state.selectionCache) return state.selectionCache;

        const nodeIds = new Set();
        const edgeIds = new Set();
        let clusterId = null;

        if (!state.selected) {
            state.selectionCacheKey = key;
            state.selectionCache = { nodeIds, edgeIds, clusterId, active: false };
            return state.selectionCache;
        }

        if (state.selected.type === "node") {
            const node = state.nodes.find((item) => item.id === state.selected.id);
            if (node) {
                nodeIds.add(node.id);
                node.edgeIds.forEach((edgeId) => {
                    const edge = state.hyperedgeById.get(edgeId);
                    if (edge) {
                        edgeIds.add(edge.id);
                        edge.members.forEach((member) => nodeIds.add(member.id));
                    }
                });
            }
        }

        if (state.selected.type === "edge") {
            const edge = state.hyperedgeById.get(state.selected.id);
            if (edge) {
                edgeIds.add(edge.id);
                edge.members.forEach((member) => nodeIds.add(member.id));
            }
        }

        if (state.selected.type === "cluster") {
            clusterId = state.selected.id;
            const cluster = state.clusters.find((item) => item.id === clusterId);
            if (cluster) {
                cluster.members.forEach((node) => nodeIds.add(node.id));
                cluster.edgeIds.forEach((edgeId) => edgeIds.add(edgeId));
            }
        }

        state.selectionCacheKey = key;
        state.selectionCache = { nodeIds, edgeIds, clusterId, active: true };
        return state.selectionCache;
    }

    function formatDistribution(rows, prefix = "") {
        if (!rows.length) return "无";
        return rows.map(([key, count]) => `${prefix}${key}: ${count}`).join("，");
    }

    function updateMetricPanel(payload) {
        if (!metricPanel) return;
        const metrics = payload.metrics || {};
        metricPanel.innerHTML = ["nmi", "ari", "acc", "f1"]
            .filter((key) => Number.isFinite(metrics[key]))
            .map((key) => `<span>${key.toUpperCase()} ${metrics[key].toFixed(4)}</span>`)
            .join("");
    }

    function updateHyperedgeSummary() {
        if (!hyperedgePanel) return;
        const visibleEdges = filteredHyperedges();
        if (!state.hyperedges.length) {
            hyperedgePanel.innerHTML = "<span>超边 0</span>";
            return;
        }
        const avgSize = visibleEdges.length ? visibleEdges.reduce((sum, edge) => sum + edge.members.length, 0) / visibleEdges.length : 0;
        const avgPurity = visibleEdges.length ? visibleEdges.reduce((sum, edge) => sum + edge.purity, 0) / visibleEdges.length : 0;
        const mixedCount = visibleEdges.filter((edge) => edge.mixed).length;
        hyperedgePanel.innerHTML = [
            `<span>显示 ${visibleEdges.length}/${state.hyperedges.length}</span>`,
            `<span>平均规模 ${avgSize.toFixed(1)}</span>`,
            `<span>平均纯度 ${avgPurity.toFixed(3)}</span>`,
            `<span>跨簇 ${mixedCount}</span>`,
        ].join("");
    }

    function renderSelectionPanel() {
        if (!selectionPanel) return;
        if (!state.selected) {
            selectionPanel.innerHTML = "<h3>当前选择</h3><p>点击画布中的节点、超边中心或簇标签，查看结构详情。</p>";
            renderRecommendations();
            return;
        }

        if (state.selected.type === "node") {
            const node = state.nodes.find((item) => item.id === state.selected.id);
            if (!node) return;
            const edges = state.hyperedges.filter((edge) => edge.members.some((member) => member.id === node.id));
            const avgPurity = edges.length ? edges.reduce((sum, edge) => sum + edge.purity, 0) / edges.length : 0;
            selectionPanel.innerHTML = `
                <h3>节点 ${node.id}</h3>
                <dl>
                    <dt>预测簇</dt><dd>C${node.cluster}</dd>
                    <dt>真实标签</dt><dd>${node.label}</dd>
                    <dt>所属超边</dt><dd>${edges.length}</dd>
                    <dt>相关超边平均纯度</dt><dd>${avgPurity.toFixed(3)}</dd>
                </dl>
            `;
            renderRecommendations(node);
        }

        if (state.selected.type === "edge") {
            const edge = state.hyperedges.find((item) => item.id === state.selected.id);
            if (!edge) return;
            selectionPanel.innerHTML = `
                <h3>超边 ${edge.id}</h3>
                <dl>
                    <dt>规模</dt><dd>${edge.members.length} 个节点</dd>
                    <dt>主导预测簇</dt><dd>C${edge.dominant}</dd>
                    <dt>纯度</dt><dd>${edge.purity.toFixed(3)}</dd>
                    <dt>是否跨簇</dt><dd>${edge.mixed ? "是" : "否"}</dd>
                    <dt>预测簇分布</dt><dd>${formatDistribution(edge.clusterCounts, "C")}</dd>
                    <dt>真实标签分布</dt><dd>${formatDistribution(edge.labelCounts)}</dd>
                </dl>
            `;
            renderRecommendations();
        }

        if (state.selected.type === "cluster") {
            const cluster = state.clusters.find((item) => item.id === state.selected.id);
            if (!cluster) return;
            selectionPanel.innerHTML = `
                <h3>簇 C${cluster.id}</h3>
                <dl>
                    <dt>簇内节点</dt><dd>${cluster.size}</dd>
                    <dt>相关超边</dt><dd>${cluster.edgeIds.length}</dd>
                    <dt>平均超边纯度</dt><dd>${cluster.avgHyperedgePurity.toFixed(3)}</dd>
                    <dt>跨簇超边</dt><dd>${cluster.mixedHyperedgeCount}</dd>
                    <dt>真实标签分布</dt><dd>${formatDistribution(cluster.labelDistribution)}</dd>
                </dl>
            `;
            renderRecommendations();
        }
    }

    function scoreRelatedNodes(sourceNode) {
        if (!sourceNode) return [];
        const sourceEdges = new Set(sourceNode.edgeIds || []);
        return state.nodes
            .filter((node) => node.id !== sourceNode.id)
            .map((node) => {
                const sharedEdges = (node.edgeIds || []).filter((edgeId) => sourceEdges.has(edgeId)).length;
                const sameCluster = node.cluster === sourceNode.cluster;
                const sameLabel = node.label === sourceNode.label;
                const score = sharedEdges * 3 + (sameCluster ? 2 : 0) + (sameLabel ? 1 : 0);
                return { node, score, sharedEdges, sameCluster, sameLabel };
            })
            .filter((item) => item.score > 0)
            .sort((a, b) => b.score - a.score || b.sharedEdges - a.sharedEdges)
            .slice(0, 5);
    }

    function renderRecommendations(sourceNode = null) {
        if (!recommendationPanel) return;
        if (!sourceNode) {
            recommendationPanel.innerHTML = "<p>选择一个节点后，系统会基于同簇关系、共享超边和标签一致性推荐相关节点。</p>";
            return;
        }
        const items = scoreRelatedNodes(sourceNode);
        if (!items.length) {
            recommendationPanel.innerHTML = "<p>当前节点没有可解释的相关节点。</p>";
            return;
        }
        recommendationPanel.innerHTML = items
            .map(({ node, score, sharedEdges, sameCluster, sameLabel }) => {
                const tags = [
                    sameCluster ? "同簇" : "",
                    sharedEdges ? `共享超边 ${sharedEdges}` : "",
                    sameLabel ? "标签一致" : "",
                ].filter(Boolean);
                return `
                    <button type="button" data-node-id="${node.id}">
                        <strong>节点 ${node.id}</strong>
                        <span>score=${score} · C${node.cluster} · L${node.label}</span>
                        <em>${tags.map((tag) => `<i>${tag}</i>`).join("")}</em>
                    </button>
                `;
            })
            .join("");
    }

    function renderTopEdges() {
        if (!topEdgeList) return;
        const edges = [...state.hyperedges].sort((a, b) => a.purity - b.purity || b.members.length - a.members.length).slice(0, 10);
        topEdgeList.innerHTML = edges
            .map((edge) => `<button type="button" data-edge-id="${edge.id}">#${edge.id}<span>size=${edge.members.length} purity=${edge.purity.toFixed(3)}</span></button>`)
            .join("");
    }

    function selectRepresentativeEdge(kind) {
        let candidates = [...state.hyperedges];
        if (kind === "mixed") {
            edgeFilter.value = "mixed";
            candidates = candidates.filter((edge) => edge.mixed).sort((a, b) => a.purity - b.purity || b.members.length - a.members.length);
        } else if (kind === "low") {
            edgeFilter.value = "low";
            candidates = candidates.filter((edge) => edge.purity < 0.55).sort((a, b) => a.purity - b.purity || b.members.length - a.members.length);
        } else if (kind === "largest") {
            edgeFilter.value = "all";
            candidates = candidates.sort((a, b) => b.members.length - a.members.length || a.purity - b.purity);
        }

        state.filteredEdgeKey = null;
        updateHyperedgeSummary();
        const edge = candidates[0];
        if (!edge) {
            setStatus("当前数据集中没有匹配的超边。");
            return;
        }
        state.selected = { type: "edge", id: edge.id };
        state.selectionCacheKey = null;
        renderSelectionPanel();
        setStatus(`已定位超边 ${edge.id}。`);
    }

    function updateHyperedgePositions() {
        state.hyperedges.forEach((edge) => {
            if (!edge.members.length) return;
            edge.x = edge.members.reduce((sum, node) => sum + node.x, 0) / edge.members.length;
            edge.y = edge.members.reduce((sum, node) => sum + node.y, 0) / edge.members.length;
        });
    }

    function drawReadyHyperedges() {
        const selection = relatedSelectionSets();
        const hoveredId = state.hoveredEdge?.id;
        const filtered = filteredHyperedges();
        const maxEdges = state.selected?.type === "cluster" ? 95 : selection.active ? 180 : 130;
        if (filtered.length <= maxEdges) return filtered;

        const important = [];
        const seen = new Set();
        const addEdge = (edge) => {
            if (!edge || seen.has(edge.id) || important.length >= maxEdges) return;
            important.push(edge);
            seen.add(edge.id);
        };

        if (hoveredId !== undefined) addEdge(state.hyperedgeById.get(hoveredId));

        if (selection.active) {
            const selectedLimit = state.selected?.type === "cluster" ? 36 : 120;
            filtered
                .filter((edge) => selection.edgeIds.has(edge.id))
                .sort((a, b) => a.purity - b.purity || b.members.length - a.members.length)
                .slice(0, selectedLimit)
                .forEach(addEdge);
            if (state.selected?.type === "cluster") return important;
        }

        for (const edge of filtered) {
            if (important.length >= maxEdges) break;
            if (!seen.has(edge.id)) important.push(edge);
        }
        return important;
    }

    function renderClusterMatrix() {
        if (!clusterMatrix) return;
        if (!state.clusters.length) {
            clusterMatrix.innerHTML = "";
            return;
        }

        const rows = [...state.clusters]
            .sort((a, b) => b.size - a.size)
            .slice(0, 8)
            .map((cluster) => {
                const labels = distribution(cluster.members, "label");
                const labelParts = labels.slice(0, 4).map(([label, count]) => {
                    const ratio = count / cluster.size;
                    return `
                        <div class="matrix-bar">
                            <span>L${label}</span>
                            <i style="width: ${Math.max(6, ratio * 100)}%"></i>
                            <b>${Math.round(ratio * 100)}%</b>
                        </div>
                    `;
                }).join("");
                return `
                    <button type="button" data-cluster-id="${cluster.id}">
                        <strong>C${cluster.id}</strong>
                        <em>${cluster.size} nodes</em>
                        ${labelParts}
                    </button>
                `;
            });

        clusterMatrix.innerHTML = rows.join("");
    }

    function applyPayload(payload) {
        state.payload = payload;
        state.elapsed = 0;
        state.lastTime = 0;
        state.hoveredNode = null;
        state.hoveredEdge = null;
        state.selected = null;
        rebuildScene();

        if (nodeCountLabel) nodeCountLabel.textContent = `${payload.shown_node_count}/${payload.node_count}`;
        if (clusterCountLabel) clusterCountLabel.textContent = String(payload.cluster_count);
        if (hyperedgeCountLabel) hyperedgeCountLabel.textContent = String((payload.hyperedges || []).length);
        if (stageLabel) stageLabel.textContent = payload.dataset;
        updateMetricPanel(payload);
        setStatus(`已加载 ${payload.dataset}：展示 ${payload.shown_node_count} / ${payload.node_count} 个节点，${(payload.hyperedges || []).length} 条超边。`);
    }

    async function loadDataset(file) {
        setStatus("正在加载超图聚类结果...");
        if (window.CLUSTER_DATASETS && window.CLUSTER_DATASETS[file]) {
            applyPayload(window.CLUSTER_DATASETS[file]);
            return;
        }
        try {
            const response = await fetch(`data/cluster/${file}`, { cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            applyPayload(await response.json());
        } catch (error) {
            state.payload = null;
            state.nodes = [];
            state.clusters = [];
            state.hyperedges = [];
            if (metricPanel) metricPanel.innerHTML = "";
            if (hyperedgePanel) hyperedgePanel.innerHTML = "";
            if (nodeCountLabel) nodeCountLabel.textContent = "--";
            if (clusterCountLabel) clusterCountLabel.textContent = "--";
            if (hyperedgeCountLabel) hyperedgeCountLabel.textContent = "--";
            if (stageLabel) stageLabel.textContent = "--";
            setStatus("没有读取到该数据集结果。请先运行训练导出命令：python submit_main.py --config survey_config.yaml --export-visualization");
        }
    }

    async function loadManifest() {
        const useManifest = (manifest) => {
            const datasets = manifest.datasets || [];
            if (!datasets.length) {
                setStatus("还没有导出的聚类结果。请先运行训练导出命令。");
                return;
            }
            datasetSelect.innerHTML = datasets.map((item) => `<option value="${item.file}">${item.name}</option>`).join("");
            const survey = datasets.find((item) => item.name === "survey_stress");
            if (survey) datasetSelect.value = survey.file;
            datasetSelect.addEventListener("change", () => loadDataset(datasetSelect.value));
            loadDataset(datasetSelect.value || datasets[0].file);
        };

        if (window.CLUSTER_DATA_MANIFEST) {
            useManifest(window.CLUSTER_DATA_MANIFEST);
            return;
        }
        try {
            const response = await fetch("data/cluster/manifest.json", { cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            useManifest(await response.json());
        } catch (error) {
            setStatus("未找到 data/cluster/manifest.json。请先运行训练导出命令。");
        }
    }

    function update(delta) {
        state.elapsed += delta;
        const clusterById = new Map(state.clusters.map((cluster) => [cluster.id, cluster]));
        state.nodes.forEach((node) => {
            const cluster = clusterById.get(node.cluster);
            const isDraggingNode = state.drag.type === "node" && state.drag.node?.id === node.id;
            const isDraggedEdgeMember = state.drag.type === "edge" && state.drag.memberOffsets.some((item) => item.node.id === node.id);
            if (isDraggingNode || isDraggedEdgeMember) {
                return;
            }
            if (node.pinned) {
                node.vx *= 0.5;
                node.vy *= 0.5;
                return;
            }
            const wobble = Math.sin(state.elapsed * 0.0016 + node.phase) * 2.8;
            const tx = node.tx + Math.cos(node.phase) * wobble;
            const ty = node.ty + Math.sin(node.phase) * wobble;
            node.vx += (tx - node.x) * 0.018;
            node.vy += (ty - node.y) * 0.018;
            if (cluster) {
                node.vx += (cluster.x - node.x) * 0.001;
                node.vy += (cluster.y - node.y) * 0.001;
            }
            node.vx *= 0.84;
            node.vy *= 0.84;
            node.x += node.vx;
            node.y += node.vy;
        });

        updateHyperedgePositions();
    }

    function drawBackground() {
        ctx.clearRect(0, 0, state.width, state.height);
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(0, 0, state.width, state.height);
        ctx.save();
        ctx.strokeStyle = "rgba(15, 23, 42, 0.045)";
        ctx.lineWidth = 1;
        const grid = 48;
        for (let x = grid; x < state.width; x += grid) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, state.height);
            ctx.stroke();
        }
        for (let y = grid; y < state.height; y += grid) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(state.width, y);
            ctx.stroke();
        }
        ctx.restore();
    }

    function edgeColor(edge) {
        if (edge.purity >= 0.8) return "#00a884";
        if (edge.purity >= 0.55) return "#f59e0b";
        return "#ef4444";
    }

    function drawHyperedges() {
        if (!showHyperedges?.checked) return;
        const selection = relatedSelectionSets();
        drawReadyHyperedges().forEach((edge) => {
            const selected = selection.edgeIds.has(edge.id);
            const faded = selection.active && !selected;
            const hovered = state.hoveredEdge && state.hoveredEdge.id === edge.id;
            const color = edgeColor(edge);
            const clusterMode = state.selected?.type === "cluster";
            const memberLimit = clusterMode ? (selected || hovered ? 8 : 4) : (selected || hovered ? 32 : 12);
            const visibleMembers = edge.members.slice(0, memberLimit);
            const alpha = selected || hovered ? (clusterMode ? 0.28 : 0.52) : faded ? 0.025 : clusterMode ? 0.055 : 0.13 + edge.purity * 0.1;

            ctx.save();
            ctx.lineWidth = clusterMode ? (selected || hovered ? 1.1 : 0.65) : selected || hovered ? 1.9 : 0.95;
            ctx.strokeStyle = hexToRgba(color, alpha);
            ctx.fillStyle = hexToRgba(color, selected || hovered ? (clusterMode ? 0.045 : 0.09) : 0.028);
            ctx.shadowColor = selected || hovered ? hexToRgba(color, 0.28) : "transparent";
            ctx.shadowBlur = clusterMode ? 0 : selected || hovered ? 14 : 0;

            const radius = Math.max(12, Math.min(34, edge.members.length * 0.55));
            ctx.beginPath();
            ctx.arc(edge.x, edge.y, radius, 0, Math.PI * 2);
            ctx.fill();

            visibleMembers.forEach((node) => {
                ctx.beginPath();
                ctx.moveTo(edge.x, edge.y);
                ctx.quadraticCurveTo((edge.x + node.x) / 2, (edge.y + node.y) / 2 - 10, node.x, node.y);
                ctx.stroke();
            });
            ctx.restore();
        });
    }

    function drawClusters() {
        const selection = relatedSelectionSets();
        state.clusters.forEach((cluster) => {
            const selected = selection.clusterId === cluster.id;
            const faded = selection.active && selection.clusterId !== null && !selected;
            const pulse = 0.96 + Math.sin(state.elapsed * 0.0016 + cluster.id) * 0.04;
            const gradient = ctx.createRadialGradient(cluster.x, cluster.y, 0, cluster.x, cluster.y, cluster.radius);
            gradient.addColorStop(0, hexToRgba(cluster.color, selected ? 0.28 : faded ? 0.035 : 0.15));
            gradient.addColorStop(0.65, hexToRgba(cluster.color, selected ? 0.1 : faded ? 0.015 : 0.055));
            gradient.addColorStop(1, hexToRgba(cluster.color, 0));
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(cluster.x, cluster.y, cluster.radius * pulse, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    function drawNodes() {
        const mode = colorMode?.value || "cluster";
        const selection = relatedSelectionSets();
        ctx.save();
        state.nodes.forEach((node) => {
            const selected = selection.nodeIds.has(node.id);
            const faded = selection.active && !selected;
            const active = selected || (state.hoveredNode && state.hoveredNode.id === node.id);
            const color = mode === "purity" ? "#64748b" : colorFor(mode === "label" ? node.label : node.cluster, mode);
            ctx.globalAlpha = active ? 1 : faded ? 0.16 : mode === "purity" ? 0.48 : 0.84;
            ctx.fillStyle = color;
            ctx.shadowColor = active ? hexToRgba(colorFor(node.cluster, "cluster"), 0.35) : "transparent";
            ctx.shadowBlur = active ? 14 : 0;
            ctx.beginPath();
            ctx.arc(node.x, node.y, active ? node.size + 2.2 : node.size, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = active ? 0.9 : faded ? 0.18 : 0.68;
            ctx.strokeStyle = node.pinned ? "#0f172a" : colorFor(node.label, "label");
            ctx.lineWidth = node.pinned ? 2 : mode === "cluster" ? 1.25 : 0.8;
            ctx.stroke();
        });
        ctx.restore();
    }

    function drawClusterLabels() {
        const previousBoxes = state.labelBoxes;
        const previousBoxByCluster = new Map(previousBoxes.map((box) => [box.clusterId, box]));
        state.labelBoxes = [];
        const rankedClusters = [...state.clusters].sort((a, b) => b.size - a.size);
        const maxLabels = state.clusters.length > 18 ? 18 : state.clusters.length;
        const visibleIds = new Set(rankedClusters.slice(0, maxLabels).map((cluster) => cluster.id));
        const selection = relatedSelectionSets();

        state.clusters.forEach((cluster, index) => {
            if (!visibleIds.has(cluster.id)) return;
            const oldBox = previousBoxByCluster.get(cluster.id);
            const selected = state.selected?.type === "cluster" && state.selected.id === cluster.id;
            const faded = selection.active && selection.clusterId !== null && !selected;
            const label = `${cluster.label} (${cluster.size})`;
            ctx.font = "800 13px Montserrat, Arial, sans-serif";
            const textWidth = ctx.measureText(label).width;
            const labelWidth = textWidth + 20;
            const labelHeight = 27;
            const targetX = Math.min(state.width - labelWidth - 12, Math.max(12, cluster.x - labelWidth / 2));
            const targetY = Math.min(state.height - labelHeight - 12, Math.max(12, cluster.y + 24 + (index % 2) * 8));
            const labelX = oldBox ? oldBox.x + (targetX - oldBox.x) * 0.18 : targetX;
            const labelY = oldBox ? oldBox.y + (targetY - oldBox.y) * 0.18 : targetY;
            state.labelBoxes.push({ clusterId: cluster.id, x: labelX, y: labelY, width: labelWidth, height: labelHeight });

            ctx.save();
            ctx.globalAlpha = faded ? 0.32 : 1;
            ctx.shadowColor = selected ? hexToRgba(cluster.color, 0.28) : "rgba(15, 23, 42, 0.12)";
            ctx.shadowBlur = selected ? 18 : 16;
            ctx.shadowOffsetY = 6;
            ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
            roundedRect(labelX, labelY, labelWidth, labelHeight, 8);
            ctx.fill();
            ctx.restore();

            ctx.save();
            ctx.globalAlpha = faded ? 0.32 : 1;
            ctx.strokeStyle = hexToRgba(cluster.color, selected ? 0.7 : 0.3);
            ctx.lineWidth = selected ? 1.6 : 1;
            roundedRect(labelX, labelY, labelWidth, labelHeight, 8);
            ctx.stroke();
            ctx.fillStyle = "#172033";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(label, labelX + labelWidth / 2, labelY + labelHeight / 2 + 1);
            ctx.restore();
        });
    }

    function drawEmptyState() {
        ctx.fillStyle = "#64748b";
        ctx.font = "700 18px Montserrat, Arial, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("等待加载超图聚类结果", state.width / 2, state.height / 2);
    }

    function nearestNode(x, y) {
        let best = null;
        let bestDistance = 14;
        state.nodes.forEach((node) => {
            const distance = Math.hypot(node.x - x, node.y - y);
            if (distance < bestDistance) {
                best = node;
                bestDistance = distance;
            }
        });
        return best;
    }

    function nearestHyperedge(x, y) {
        if (!showHyperedges?.checked) return null;
        let best = null;
        let bestDistance = 28;
        filteredHyperedges().forEach((edge) => {
            const distance = Math.hypot(edge.x - x, edge.y - y);
            if (distance < bestDistance) {
                best = edge;
                bestDistance = distance;
            }
        });
        return best;
    }

    function hitClusterLabel(x, y) {
        return state.labelBoxes.find((box) => x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height);
    }

    function showTooltip(html, x, y) {
        if (!tooltip) return;
        tooltip.innerHTML = html;
        tooltip.style.left = `${Math.min(state.width - 230, x + 18)}px`;
        tooltip.style.top = `${Math.max(12, y + 18)}px`;
        tooltip.classList.add("show");
    }

    function hideTooltip() {
        tooltip?.classList.remove("show");
    }

    function handlePointerMove(event) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const labelHit = hitClusterLabel(x, y);
        state.hoveredNode = labelHit ? null : nearestNode(x, y);
        state.hoveredEdge = state.hoveredNode || labelHit ? null : nearestHyperedge(x, y);

        if (labelHit) {
            showTooltip(`<strong>簇 C${labelHit.clusterId}</strong><br>点击查看簇内真实标签分布`, x, y);
        } else if (state.hoveredNode) {
            showTooltip(`<strong>节点 ${state.hoveredNode.id}</strong><br>预测簇：C${state.hoveredNode.cluster}<br>真实标签：${state.hoveredNode.label}`, x, y);
        } else if (state.hoveredEdge) {
            const clusters = state.hoveredEdge.clusterCounts.map(([cluster, count]) => `C${cluster}:${count}`).join(" ");
            showTooltip(`<strong>超边 ${state.hoveredEdge.id}</strong><br>规模：${state.hoveredEdge.members.length} 个节点<br>主导簇：C${state.hoveredEdge.dominant}<br>纯度：${state.hoveredEdge.purity.toFixed(3)}<br>${clusters}`, x, y);
        } else {
            hideTooltip();
        }
    }

    function handleCanvasClick(event) {
        if (state.drag.moved) return;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const labelHit = hitClusterLabel(x, y);
        const node = labelHit ? null : nearestNode(x, y);
        const edge = labelHit || node ? null : nearestHyperedge(x, y);

        if (labelHit) state.selected = { type: "cluster", id: labelHit.clusterId };
        else if (node) state.selected = { type: "node", id: node.id };
        else if (edge) state.selected = { type: "edge", id: edge.id };
        else state.selected = null;

        state.selectionCacheKey = null;
        renderSelectionPanel();
    }

    function handlePointerDown(event) {
        if (event.button !== undefined && event.button !== 0) return;
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const node = nearestNode(x, y);
        const edge = node ? null : nearestHyperedge(x, y);
        if (!node && !edge) return;
        event.preventDefault();
        state.drag.type = node ? "node" : "edge";
        state.drag.node = node;
        state.drag.edge = edge;
        state.drag.moved = false;
        state.drag.pointerId = event.pointerId;
        state.drag.startX = x;
        state.drag.startY = y;
        state.drag.offsetX = (node ? node.x : edge.x) - x;
        state.drag.offsetY = (node ? node.y : edge.y) - y;
        state.drag.targetX = node ? node.x : edge.x;
        state.drag.targetY = node ? node.y : edge.y;
        state.drag.lastX = node ? node.x : edge.x;
        state.drag.lastY = node ? node.y : edge.y;
        state.drag.memberOffsets = edge
            ? edge.members.map((member) => ({
                  node: member,
                  dx: member.x - edge.x,
                  dy: member.y - edge.y,
              }))
            : [];
        if (node) {
            node.pinned = true;
            node.vx = 0;
            node.vy = 0;
            state.selected = { type: "node", id: node.id };
        } else {
            edge.members.forEach((member) => {
                member.pinned = true;
                member.vx = 0;
                member.vy = 0;
            });
            state.selected = { type: "edge", id: edge.id };
        }
        renderSelectionPanel();
        canvas.setPointerCapture?.(event.pointerId);
        canvas.classList.add("is-dragging", state.drag.type === "edge" ? "is-dragging-edge" : "is-dragging-node");
    }

    function handlePointerDrag(event) {
        if (!state.drag.node && !state.drag.edge) {
            handlePointerMove(event);
            return;
        }
        if (state.drag.pointerId !== null && event.pointerId !== state.drag.pointerId) return;
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const nextX = x + state.drag.offsetX;
        const nextY = y + state.drag.offsetY;
        if (Math.hypot(x - state.drag.startX, y - state.drag.startY) > 4) state.drag.moved = true;
        state.drag.targetX = Math.max(10, Math.min(state.width - 10, nextX));
        state.drag.targetY = Math.max(10, Math.min(state.height - 10, nextY));

        if (state.drag.type === "node" && state.drag.node) {
            const node = state.drag.node;
            node.x += (state.drag.targetX - node.x) * 0.72;
            node.y += (state.drag.targetY - node.y) * 0.72;
            node.tx = node.x;
            node.ty = node.y;
            node.vx = 0;
            node.vy = 0;
            state.hoveredNode = node;
            state.hoveredEdge = null;
        }

        if (state.drag.type === "edge" && state.drag.edge) {
            const deltaX = state.drag.targetX - state.drag.lastX;
            const deltaY = state.drag.targetY - state.drag.lastY;
            state.drag.memberOffsets.forEach(({ node }) => {
                const targetX = Math.max(10, Math.min(state.width - 10, node.x + deltaX));
                const targetY = Math.max(10, Math.min(state.height - 10, node.y + deltaY));
                node.x += (targetX - node.x) * 0.72;
                node.y += (targetY - node.y) * 0.72;
                node.vx = 0;
                node.vy = 0;
                node.tx = node.x;
                node.ty = node.y;
                node.pinned = true;
            });
            state.drag.lastX = state.drag.targetX;
            state.drag.lastY = state.drag.targetY;
            state.hoveredNode = null;
            state.hoveredEdge = state.drag.edge;
        }
        hideTooltip();
    }

    function handlePointerUp(event) {
        if (state.drag.pointerId !== null && event.pointerId !== state.drag.pointerId) return;
        if (state.drag.type === "node" && state.drag.node && state.drag.moved) {
            state.drag.node.x = state.drag.targetX;
            state.drag.node.y = state.drag.targetY;
            state.drag.node.tx = state.drag.targetX;
            state.drag.node.ty = state.drag.targetY;
            state.drag.node.vx = 0;
            state.drag.node.vy = 0;
            state.selected = { type: "node", id: state.drag.node.id };
            renderSelectionPanel();
            setStatus(`已固定节点 ${state.drag.node.id}，点击“重排布局”可释放手动位置。`);
        }
        if (state.drag.type === "edge" && state.drag.edge && state.drag.moved) {
            state.drag.memberOffsets.forEach(({ node }) => {
                node.x = Math.max(10, Math.min(state.width - 10, node.x));
                node.y = Math.max(10, Math.min(state.height - 10, node.y));
                node.tx = node.x;
                node.ty = node.y;
                node.vx = 0;
                node.vy = 0;
                node.pinned = true;
            });
            state.selected = { type: "edge", id: state.drag.edge.id };
            state.selectionCacheKey = null;
            renderSelectionPanel();
            setStatus(`已整体移动超边 ${state.drag.edge.id} 及其节点，点击“重排布局”可释放手动位置。`);
        }
        if (state.drag.pointerId !== null) canvas.releasePointerCapture?.(state.drag.pointerId);
        state.drag.type = null;
        state.drag.node = null;
        state.drag.edge = null;
        state.drag.pointerId = null;
        state.drag.memberOffsets = [];
        state.drag.settleUntil = 0;
        updateHyperedgePositions();
        canvas.classList.remove("is-dragging", "is-dragging-node", "is-dragging-edge");
        setTimeout(() => {
            state.drag.moved = false;
        }, 0);
    }

    function render(time) {
        const delta = state.lastTime ? time - state.lastTime : 16;
        state.lastTime = time;
        if (state.nodes.length) update(Math.min(delta, 42));
        drawBackground();
        if (state.nodes.length) {
            drawHyperedges();
            drawClusters();
            drawNodes();
            drawClusterLabels();
        } else {
            drawEmptyState();
        }
        requestAnimationFrame(render);
    }

    relayoutButton?.addEventListener("click", () => {
        state.nodes.forEach((node) => {
            node.pinned = false;
        });
        rebuildScene();
    });

    resetFocusButton?.addEventListener("click", () => {
        state.selected = null;
        state.selectionCacheKey = null;
        state.nodes.forEach((node) => {
            node.pinned = false;
        });
        renderSelectionPanel();
        hideTooltip();
    });

    exportButton?.addEventListener("click", () => {
        const link = document.createElement("a");
        link.download = `intrahc-${state.payload?.dataset || "cluster"}-view.png`;
        link.href = canvas.toDataURL("image/png");
        link.click();
    });

    surveyViewButton?.addEventListener("click", () => {
        const option = Array.from(datasetSelect.options).find((item) => item.textContent === "survey_stress");
        if (!option) return;
        datasetSelect.value = option.value;
        edgeFilter.value = "mixed";
        loadDataset(option.value);
    });

    searchNodeButton?.addEventListener("click", () => {
        const id = Number(nodeSearch.value);
        const node = state.nodes.find((item) => item.id === id);
        if (!node) {
            setStatus(`未找到节点 ${nodeSearch.value}`);
            return;
        }
        state.selected = { type: "node", id: node.id };
        state.selectionCacheKey = null;
        renderSelectionPanel();
        setStatus(`已定位节点 ${node.id}`);
    });

    topEdgeList?.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-edge-id]");
        if (!button) return;
        state.selected = { type: "edge", id: Number(button.dataset.edgeId) };
        state.selectionCacheKey = null;
        renderSelectionPanel();
    });

    recommendationPanel?.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-node-id]");
        if (!button) return;
        state.selected = { type: "node", id: Number(button.dataset.nodeId) };
        state.selectionCacheKey = null;
        renderSelectionPanel();
    });

    analysisChips?.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-action]");
        if (!button) return;
        selectRepresentativeEdge(button.dataset.action);
    });

    clusterMatrix?.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-cluster-id]");
        if (!button) return;
        const clusterId = Number(button.dataset.clusterId);
        requestAnimationFrame(() => {
            state.selected = { type: "cluster", id: clusterId };
            state.selectionCacheKey = null;
            renderSelectionPanel();
        });
    });

    colorMode?.addEventListener("change", hideTooltip);
    edgeFilter?.addEventListener("change", () => {
        state.filteredEdgeKey = null;
        updateHyperedgeSummary();
        hideTooltip();
    });
    showHyperedges?.addEventListener("change", hideTooltip);
    canvas.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("pointermove", handlePointerDrag);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);
    canvas.addEventListener("pointermove", handlePointerMove);
    canvas.addEventListener("click", handleCanvasClick);
    canvas.addEventListener("mouseleave", () => {
        state.hoveredNode = null;
        state.hoveredEdge = null;
        hideTooltip();
    });

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    loadManifest();
    requestAnimationFrame(render);
});
