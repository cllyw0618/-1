(function () {
  const DATASET_FILES = {
    Cora: 'demo_data/Cora.json',
    Citeseer: 'demo_data/Citeseer.json',
    Pubmed: 'demo_data/Pubmed.json'
  };

  const CLUSTER_COLORS = [
    '#4e79ff',
    '#12b89a',
    '#8a67ff',
    '#ff9b40',
    '#ed5f9f',
    '#1dced8',
    '#607b9a',
    '#5ec25b'
  ];

  const selectorBtn = document.getElementById('datasetSelectorBtn');
  const selectorName = document.getElementById('selectorName');
  const dropdown = document.getElementById('datasetDropdown');
  const options = Array.from(document.querySelectorAll('.dataset-option'));
  const targetSection = document.getElementById('demo-showcase');

  const searchInputEl = document.getElementById('demoSearchInput');
  const searchButtonEl = document.getElementById('demoSearchButton');
  const exampleChipsEl = document.getElementById('demoExampleChips');
  const currentQueryEl = document.getElementById('demoCurrentQuery');
  const clusterNameEl = document.getElementById('demoClusterName');
  const explanationEl = document.getElementById('demoExplanation');
  const datasetNameEl = document.getElementById('demoDatasetName');
  const nodeTypeEl = document.getElementById('demoNodeType');
  const recommendModeEl = document.getElementById('demoRecommendMode');
  const visualTitleEl = document.getElementById('demoVisualTitle');
  const scatterStageEl = document.getElementById('demoScatterStage');
  const legendEl = document.getElementById('demoLegend');
  const highlightBtnEl = document.getElementById('demoHighlightBtn');
  const embeddingGridEl = document.getElementById('demoEmbeddingGrid');
  const hyperedgeGridEl = document.getElementById('demoHyperedgeGrid');

  const metricNmiEl = document.getElementById('metricNmi');
  const metricAccEl = document.getElementById('metricAcc');
  const metricAriEl = document.getElementById('metricAri');
  const metricF1El = document.getElementById('metricF1');

  const state = {
    datasetKey: 'Cora',
    data: null,
    focusNodeId: null,
    relatedNodeIds: new Set(),
    highlightRelated: true
  };

  function smoothToShowcase() {
    if (!targetSection) return;
    targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function setDropdown(open) {
    if (!dropdown || !selectorBtn) return;
    const willOpen = typeof open === 'boolean' ? open : !dropdown.classList.contains('show');
    dropdown.classList.toggle('show', willOpen);
    selectorBtn.setAttribute('aria-expanded', String(willOpen));
  }

  function normalizePercent(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0.5;
    if (n > 1) return Math.max(0, Math.min(1, n / 100));
    return Math.max(0, Math.min(1, n));
  }

  function formatNodeLabel(datasetKey, nodeId) {
    return datasetKey + ' Node #' + String(nodeId).padStart(4, '0');
  }

  function getClusterColor(clusterId) {
    const idx = Math.abs(Number(clusterId) || 0) % CLUSTER_COLORS.length;
    return CLUSTER_COLORS[idx];
  }

  function toNodeMap(nodes) {
    const map = new Map();
    nodes.forEach((node) => {
      map.set(node.id, node);
    });
    return map;
  }

  function normalizeNode(rawNode) {
    const id = Number(rawNode.id);
    const cluster = Number(rawNode.cluster);
    return {
      id,
      cluster,
      x: normalizePercent(rawNode.x),
      y: normalizePercent(rawNode.y),
      embedding_similar: Array.isArray(rawNode.embedding_similar) ? rawNode.embedding_similar : [],
      hyperedge_neighbors: Array.isArray(rawNode.hyperedge_neighbors) ? rawNode.hyperedge_neighbors : []
    };
  }

  function normalizeDataset(datasetKey, raw) {
    const nodes = Array.isArray(raw.nodes) ? raw.nodes.map(normalizeNode) : [];
    const uniqueClusters = Array.from(new Set(nodes.map((node) => node.cluster))).sort((a, b) => a - b);
    const clusterRepresentatives = raw.cluster_representatives || {};
    const defaultNodeId = Number(raw.default_node_id);

    return {
      dataset_key: datasetKey,
      node_type: raw.node_type || '匿名样本节点',
      recommend_mode: raw.recommend_mode || '同簇 + 表征相似 + 高阶关系',
      example_queries: Array.isArray(raw.example_queries) && raw.example_queries.length > 0
        ? raw.example_queries
        : ['Node 0', 'Node 128', 'Node 512', 'Cluster 3'],
      metrics: raw.metrics || {},
      nodes,
      clusters: uniqueClusters,
      cluster_representatives: clusterRepresentatives,
      default_node_id: Number.isFinite(defaultNodeId) ? defaultNodeId : (nodes[0] ? nodes[0].id : 0),
      top_k: Number(raw.top_k) > 0 ? Number(raw.top_k) : 4
    };
  }

  async function loadDataset(datasetKey) {
    const file = DATASET_FILES[datasetKey];
    if (!file) return;

    const response = await fetch(file, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Failed to load ' + file + ': ' + response.status);
    }

    const raw = await response.json();
    const normalized = normalizeDataset(datasetKey, raw);
    state.datasetKey = datasetKey;
    state.data = normalized;

    if (selectorName) selectorName.textContent = datasetKey;
    if (datasetNameEl) datasetNameEl.textContent = datasetKey;
    if (nodeTypeEl) nodeTypeEl.textContent = normalized.node_type;
    if (recommendModeEl) recommendModeEl.textContent = normalized.recommend_mode;
    if (visualTitleEl) visualTitleEl.textContent = datasetKey + ' 降维聚类可视化';

    if (searchInputEl) {
      searchInputEl.placeholder = '输入节点编号，例如 0、128、512';
      searchInputEl.value = 'Node ' + normalized.default_node_id;
    }

    if (highlightBtnEl) {
      state.highlightRelated = true;
      highlightBtnEl.textContent = '隐藏相关高亮';
    }

    renderMetrics(normalized.metrics);
    renderExampleChips(normalized.example_queries);
    runNodeQuery(normalized.default_node_id);

    options.forEach((option) => {
      option.classList.toggle('active', option.dataset.dataset === datasetKey);
    });
  }

  function renderMetrics(metrics) {
    if (metricNmiEl) metricNmiEl.textContent = metrics.nmi || '--';
    if (metricAccEl) metricAccEl.textContent = metrics.acc || '--';
    if (metricAriEl) metricAriEl.textContent = metrics.ari || '--';
    if (metricF1El) metricF1El.textContent = metrics.f1 || '--';
  }

  function renderExampleChips(chips) {
    if (!exampleChipsEl) return;
    exampleChipsEl.innerHTML = '';

    chips.forEach((chipText) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'demo-chip';
      chip.textContent = chipText;
      chip.addEventListener('click', () => {
        if (searchInputEl) searchInputEl.value = chipText;
        executeSearch(chipText);
      });
      exampleChipsEl.appendChild(chip);
    });
  }

  function parseSearchInput(raw) {
    const value = String(raw || '').trim();
    if (!value) return null;

    let match = value.match(/^cluster\s*#?\s*(\d+)$/i);
    if (match) {
      return { type: 'cluster', id: Number(match[1]) };
    }

    match = value.match(/^node\s*#?\s*(\d+)$/i);
    if (match) {
      return { type: 'node', id: Number(match[1]) };
    }

    if (/^\d+$/.test(value)) {
      return { type: 'node', id: Number(value) };
    }

    return null;
  }

  function executeSearch(rawInput) {
    if (!state.data) return;
    const parsed = parseSearchInput(rawInput);
    if (!parsed) {
      runNodeQuery(state.data.default_node_id);
      return;
    }

    if (parsed.type === 'cluster') {
      runClusterQuery(parsed.id);
      return;
    }

    runNodeQuery(parsed.id);
  }

  function buildRelatedSets(node, nodeMap) {
    const topK = state.data ? state.data.top_k : 4;

    const embedding = node.embedding_similar
      .map((item) => ({
        id: Number(item.id),
        score: Number(item.score),
        reason: item.reason || '在训练后的节点嵌入空间中距离更近'
      }))
      .filter((item) => nodeMap.has(item.id))
      .slice(0, topK);

    const hyperedge = node.hyperedge_neighbors
      .map((item) => ({
        id: Number(item.id),
        shared_hyperedges: Number(item.shared_hyperedges),
        reason: item.reason || '与查询节点共享更多超边结构上下文'
      }))
      .filter((item) => nodeMap.has(item.id))
      .slice(0, topK);

    return { embedding, hyperedge };
  }

  function renderRelatedCards(items, type, containerEl) {
    if (!containerEl) return;
    containerEl.innerHTML = '';

    if (!state.data) return;
    const nodeMap = toNodeMap(state.data.nodes);

    if (!items.length) {
      const empty = document.createElement('article');
      empty.className = 'demo-related-card';
      empty.innerHTML = '<h4>暂无可展示节点</h4><p class="demo-related-empty">该类别下当前没有可用结果。</p>';
      containerEl.appendChild(empty);
      return;
    }

    items.forEach((item) => {
      const node = nodeMap.get(item.id);
      if (!node) return;

      const card = document.createElement('article');
      card.className = 'demo-related-card';

      const valueLine = type === 'embedding'
        ? '相似度：<strong>' + (Number.isFinite(item.score) ? item.score.toFixed(3) : '--') + '</strong>'
        : '共享超边数：<strong>' + (Number.isFinite(item.shared_hyperedges) ? item.shared_hyperedges : '--') + '</strong>';

      card.innerHTML =
        '<h4>' + formatNodeLabel(state.data.dataset_key, node.id) + '</h4>' +
        '<div class="demo-related-meta">' +
          '<span>所属簇：<strong>Cluster ' + node.cluster + '</strong></span>' +
          '<span>' + valueLine + '</span>' +
        '</div>' +
        '<p class="demo-related-reason">推荐原因：' + item.reason + '</p>';

      containerEl.appendChild(card);
    });
  }

  function renderScatter(highlightClusterId) {
    if (!scatterStageEl || !legendEl || !state.data) return;

    scatterStageEl.innerHTML = '';

    const shouldFadeByCluster = Number.isFinite(highlightClusterId);

    state.data.nodes.forEach((node) => {
      const dot = document.createElement('span');
      dot.className = 'demo-scatter-dot dynamic';
      dot.style.left = (node.x * 100).toFixed(2) + '%';
      dot.style.top = (node.y * 100).toFixed(2) + '%';
      dot.style.backgroundColor = getClusterColor(node.cluster);

      if (shouldFadeByCluster && node.cluster !== highlightClusterId) {
        dot.classList.add('faded');
      }

      if (state.highlightRelated && state.relatedNodeIds.has(node.id)) {
        dot.classList.add('related');
      }

      if (state.focusNodeId === node.id) {
        dot.classList.add('active');
      }

      dot.title = formatNodeLabel(state.data.dataset_key, node.id) + ' | Cluster ' + node.cluster;
      scatterStageEl.appendChild(dot);
    });

    legendEl.innerHTML = '';
    state.data.clusters.forEach((clusterId) => {
      const item = document.createElement('span');
      item.innerHTML = '<i class="lg" style="background:' + getClusterColor(clusterId) + '"></i>Cluster ' + clusterId;
      legendEl.appendChild(item);
    });

    const currentItem = document.createElement('span');
    currentItem.innerHTML = '<i class="lg active"></i>当前查询节点';
    legendEl.appendChild(currentItem);

    const relatedItem = document.createElement('span');
    relatedItem.innerHTML = '<i class="lg related"></i>相关节点';
    legendEl.appendChild(relatedItem);
  }

  function runNodeQuery(nodeId) {
    if (!state.data) return;
    const nodeMap = toNodeMap(state.data.nodes);

    let node = nodeMap.get(Number(nodeId));
    if (!node) {
      node = nodeMap.get(state.data.default_node_id) || state.data.nodes[0];
    }
    if (!node) return;

    state.focusNodeId = node.id;

    if (currentQueryEl) currentQueryEl.textContent = formatNodeLabel(state.data.dataset_key, node.id);
    if (clusterNameEl) clusterNameEl.textContent = '所属聚类：Cluster ' + node.cluster;
    if (explanationEl) {
      explanationEl.textContent =
        '该节点被模型分配至 Cluster ' + node.cluster +
        '。系统基于训练后的节点嵌入、预测簇标签与超边结构，返回其相似节点和高阶关系邻居。';
    }

    const related = buildRelatedSets(node, nodeMap);
    state.relatedNodeIds = new Set([
      ...related.embedding.map((item) => item.id),
      ...related.hyperedge.map((item) => item.id)
    ]);

    renderRelatedCards(related.embedding, 'embedding', embeddingGridEl);
    renderRelatedCards(related.hyperedge, 'hyperedge', hyperedgeGridEl);
    renderScatter();
  }

  function runClusterQuery(clusterId) {
    if (!state.data) return;

    const numericCluster = Number(clusterId);
    const clusterNodes = state.data.nodes.filter((node) => node.cluster === numericCluster);
    if (!clusterNodes.length) {
      runNodeQuery(state.data.default_node_id);
      return;
    }

    const explicitRepresentative = Number(state.data.cluster_representatives[String(numericCluster)]);
    const representative = clusterNodes.find((node) => node.id === explicitRepresentative) || clusterNodes[0];

    state.focusNodeId = representative.id;

    if (currentQueryEl) currentQueryEl.textContent = state.data.dataset_key + ' Cluster ' + numericCluster;
    if (clusterNameEl) clusterNameEl.textContent = '所属聚类：Cluster ' + numericCluster;
    if (explanationEl) {
      explanationEl.textContent =
        '当前查询为 Cluster ' + numericCluster +
        '。系统展示该簇代表节点 ' + formatNodeLabel(state.data.dataset_key, representative.id) +
        '，并返回簇内高相关节点与共享超边邻居。';
    }

    const nodeMap = toNodeMap(state.data.nodes);
    const related = buildRelatedSets(representative, nodeMap);
    state.relatedNodeIds = new Set([
      ...clusterNodes.map((node) => node.id),
      ...related.embedding.map((item) => item.id),
      ...related.hyperedge.map((item) => item.id)
    ]);

    renderRelatedCards(related.embedding, 'embedding', embeddingGridEl);
    renderRelatedCards(related.hyperedge, 'hyperedge', hyperedgeGridEl);
    renderScatter(numericCluster);
  }

  if (selectorBtn) {
    selectorBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      setDropdown();
      smoothToShowcase();
    });
  }

  options.forEach((option) => {
    option.addEventListener('click', async (event) => {
      event.stopPropagation();
      const chosen = option.dataset.dataset;
      if (!chosen) return;

      try {
        await loadDataset(chosen);
      } catch (error) {
        console.error(error);
      }

      setDropdown(false);
      smoothToShowcase();
    });
  });

  if (searchButtonEl && searchInputEl) {
    searchButtonEl.addEventListener('click', () => {
      executeSearch(searchInputEl.value);
    });
  }

  if (searchInputEl) {
    searchInputEl.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      executeSearch(searchInputEl.value);
    });
  }

  if (highlightBtnEl) {
    highlightBtnEl.addEventListener('click', () => {
      state.highlightRelated = !state.highlightRelated;
      highlightBtnEl.textContent = state.highlightRelated ? '隐藏相关高亮' : '高亮相关节点';
      renderScatter();
    });
  }

  document.addEventListener('click', (event) => {
    if (!dropdown || !selectorBtn) return;
    if (!dropdown.contains(event.target) && !selectorBtn.contains(event.target)) {
      setDropdown(false);
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      setDropdown(false);
    }
  });

  loadDataset(state.datasetKey).catch((error) => {
    console.error(error);
  });
})();
