(function () {
  const demoData = {
    Cora: {
      nodeType: '论文',
      searchPlaceholder: '输入论文标题、关键词或研究主题',
      examples: ['graph learning', 'semi-supervised learning', 'citation network'],
      query: 'graph learning',
      cluster: 'Cluster 3：Graph Learning / Citation Network',
      explanation: '该对象被模型识别为图学习相关主题，并与同簇论文在结构关系和语义表征上更接近。',
      recommendMode: '同簇 + 表征相似 + 高阶关系',
      visualTitle: 'Cora 降维聚类可视化',
      activePoint: { x: '58%', y: '47%' },
      reasonKeywords: ['Graph Learning', '语义嵌入', '引用超边', '主题邻近'],
      recommendations: [
        {
          title: 'Graph-Based Semi-Supervised Learning',
          cluster: 'Graph Learning',
          score: '0.92',
          tags: ['同簇', '引用邻近', '表征相似']
        },
        {
          title: 'Label Propagation in Citation Networks',
          cluster: 'Graph Learning',
          score: '0.88',
          tags: ['结构邻近', '共享超边']
        },
        {
          title: 'Robust GNN for Sparse Graphs',
          cluster: 'Representation Learning',
          score: '0.86',
          tags: ['语义相近', '边权一致']
        },
        {
          title: 'Attributed Hypergraph Contrastive Clustering',
          cluster: 'Hypergraph Methods',
          score: '0.84',
          tags: ['高阶关系', '同主题']
        },
        {
          title: 'Self-Training for Node Classification',
          cluster: 'Semi-supervised',
          score: '0.81',
          tags: ['训练策略近似', '同簇边界']
        }
      ],
      metrics: { nmi: '88.7%', acc: '92.4%', ari: '85.9%', f1: '90.1%' }
    },
    Citeseer: {
      nodeType: '论文',
      searchPlaceholder: '输入论文标题、关键词或研究主题',
      examples: ['representation learning', 'graph attention', 'citation analysis'],
      query: 'representation learning',
      cluster: 'Cluster 2：Representation Learning / Citation Analysis',
      explanation: '该查询对象位于表示学习主题簇中心区域，和多篇高被引论文共享语义与结构上下文。',
      recommendMode: '同簇 + 局部密度 + 表征相似',
      visualTitle: 'Citeseer 降维聚类可视化',
      activePoint: { x: '44%', y: '39%' },
      reasonKeywords: ['Representation Learning', '局部邻域', '高阶连接', '关键词共现'],
      recommendations: [
        {
          title: 'Attention-Based Citation Embedding',
          cluster: 'Representation Learning',
          score: '0.90',
          tags: ['同簇', '语义相近']
        },
        {
          title: 'Contextual Node Representation',
          cluster: 'Graph Embedding',
          score: '0.87',
          tags: ['嵌入相似', '结构邻近']
        },
        {
          title: 'Multi-View Citation Modeling',
          cluster: 'Citation Analysis',
          score: '0.85',
          tags: ['多视角一致', '共享超边']
        },
        {
          title: 'Semi-Supervised Graph Regularization',
          cluster: 'Graph Learning',
          score: '0.82',
          tags: ['训练机制相近', '边界接近']
        },
        {
          title: 'Contrastive Topic Clustering',
          cluster: 'Topic Mining',
          score: '0.80',
          tags: ['主题邻近', '高阶关系']
        }
      ],
      metrics: { nmi: '84.2%', acc: '89.6%', ari: '81.5%', f1: '87.3%' }
    },
    Pubmed: {
      nodeType: '论文',
      searchPlaceholder: '输入论文标题、关键词或研究主题',
      examples: ['gene expression', 'clinical trial', 'biomedical graph'],
      query: 'gene expression',
      cluster: 'Cluster 1：Biomedical Graph / Clinical Topics',
      explanation: '该对象与生物医学主题簇具有高一致性，并在高阶结构中与临床相关论文形成稳定连接。',
      recommendMode: '同簇 + 高阶关系 + 语义邻近',
      visualTitle: 'Pubmed 降维聚类可视化',
      activePoint: { x: '51%', y: '56%' },
      reasonKeywords: ['Biomedical Topic', '表征距离', '临床超边', '语义共现'],
      recommendations: [
        {
          title: 'Gene Interaction Representation in Graphs',
          cluster: 'Biomedical Graph',
          score: '0.91',
          tags: ['同簇', '主题重合']
        },
        {
          title: 'Clinical Document Graph Clustering',
          cluster: 'Clinical Topic',
          score: '0.88',
          tags: ['结构邻近', '表征相似']
        },
        {
          title: 'Disease Topic Mining with Hypergraphs',
          cluster: 'Biomedical Graph',
          score: '0.86',
          tags: ['高阶结构', '语义接近']
        },
        {
          title: 'Patient Trial Knowledge Network',
          cluster: 'Clinical Analysis',
          score: '0.83',
          tags: ['共享实体', '同簇边界']
        },
        {
          title: 'Contrastive Biomedical Embedding',
          cluster: 'Representation Learning',
          score: '0.81',
          tags: ['嵌入相似', '语义邻近']
        }
      ],
      metrics: { nmi: '79.8%', acc: '86.1%', ari: '76.9%', f1: '84.4%' }
    },
    DBLP: {
      nodeType: '作者 / 论文',
      searchPlaceholder: '输入作者姓名、论文标题或研究方向',
      examples: ['Jiawei Han', 'data mining', 'graph representation'],
      query: 'Jiawei Han',
      cluster: 'Research Community：Data Mining / Knowledge Discovery',
      explanation: '系统判断当前查询位于数据挖掘研究社区核心区域，和多个高相关作者在合作与研究主题上呈现聚合。',
      recommendMode: '研究社区 + 合作关系 + 主题相似',
      visualTitle: 'DBLP 降维聚类可视化',
      activePoint: { x: '63%', y: '41%' },
      reasonKeywords: ['Data Mining', '合作网络', '高阶协作', '研究主题'],
      recommendations: [
        {
          title: 'Philip S. Yu',
          cluster: 'Data Mining Community',
          score: '0.93',
          tags: ['合作邻近', '主题重叠']
        },
        {
          title: 'Mining Frequent Patterns in Graph Data',
          cluster: 'Knowledge Discovery',
          score: '0.90',
          tags: ['同社区', '语义相似']
        },
        {
          title: 'Charu C. Aggarwal',
          cluster: 'Graph Mining',
          score: '0.87',
          tags: ['合作路径', '共享关键词']
        },
        {
          title: 'Scalable Representation for Heterogeneous Graphs',
          cluster: 'Graph Representation',
          score: '0.84',
          tags: ['结构邻近', '高阶关系']
        },
        {
          title: 'Discovering Research Communities',
          cluster: 'Community Mining',
          score: '0.82',
          tags: ['社区相近', '协作关系']
        }
      ],
      metrics: { nmi: '82.9%', acc: '88.3%', ari: '79.4%', f1: '86.0%' }
    },
    ACM: {
      nodeType: '作者 / 论文',
      searchPlaceholder: '输入作者姓名、论文标题或研究方向',
      examples: ['Jiawei Han', 'data mining', 'graph representation'],
      query: 'data mining',
      cluster: 'Research Community：Information Retrieval / Data Mining',
      explanation: '该对象被分配到信息检索与数据挖掘交叉簇，系统综合主题相似与引用关系给出推荐。',
      recommendMode: '同社区 + 主题相似 + 高阶关系',
      visualTitle: 'ACM 降维聚类可视化',
      activePoint: { x: '47%', y: '52%' },
      reasonKeywords: ['Information Retrieval', '语义聚类', '引用超边', '研究方向'],
      recommendations: [
        {
          title: 'Topic-Aware Graph Retrieval',
          cluster: 'Information Retrieval',
          score: '0.91',
          tags: ['同簇', '语义接近']
        },
        {
          title: 'Efficient Data Mining Pipelines',
          cluster: 'Data Mining',
          score: '0.88',
          tags: ['主题相近', '结构邻近']
        },
        {
          title: 'Mining Semantic Graph Patterns',
          cluster: 'Graph Mining',
          score: '0.85',
          tags: ['高阶关系', '关键词共现']
        },
        {
          title: 'Collaborative Author Ranking',
          cluster: 'Research Community',
          score: '0.83',
          tags: ['合作关系', '同社区']
        },
        {
          title: 'Cross-Domain Representation Learning',
          cluster: 'Representation',
          score: '0.80',
          tags: ['嵌入相似', '主题邻近']
        }
      ],
      metrics: { nmi: '86.4%', acc: '90.2%', ari: '83.6%', f1: '88.8%' }
    }
  };

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
  const relatedGridEl = document.getElementById('demoRelatedGrid');
  const reasonKeyEls = [
    document.getElementById('demoReasonKey1'),
    document.getElementById('demoReasonKey2'),
    document.getElementById('demoReasonKey3'),
    document.getElementById('demoReasonKey4')
  ];
  const metricNmiEl = document.getElementById('metricNmi');
  const metricAccEl = document.getElementById('metricAcc');
  const metricAriEl = document.getElementById('metricAri');
  const metricF1El = document.getElementById('metricF1');

  let currentKey = 'Cora';

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

  function renderRecommendations(items) {
    if (!relatedGridEl) return;
    relatedGridEl.innerHTML = '';
    items.forEach((item) => {
      const card = document.createElement('article');
      card.className = 'demo-related-card';
      card.innerHTML = `
        <h4>${item.title}</h4>
        <div class="demo-related-meta">
          <span>所属簇：<strong>${item.cluster}</strong></span>
          <span>相似度：<strong>${item.score}</strong></span>
        </div>
        <div class="demo-related-tags">
          ${item.tags.map((tag) => `<span>${tag}</span>`).join('')}
        </div>
      `;
      relatedGridEl.appendChild(card);
    });
  }

  function renderExampleChips(payload) {
    if (!exampleChipsEl || !searchInputEl || !currentQueryEl) return;
    exampleChipsEl.innerHTML = '';
    payload.examples.forEach((keyword) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'demo-chip';
      chip.textContent = keyword;
      chip.addEventListener('click', () => {
        searchInputEl.value = keyword;
        currentQueryEl.textContent = keyword;
      });
      exampleChipsEl.appendChild(chip);
    });
  }

  function renderDataset(key) {
    const payload = demoData[key];
    if (!payload) return;

    currentKey = key;
    if (selectorName) selectorName.textContent = key;
    if (searchInputEl) {
      searchInputEl.placeholder = payload.searchPlaceholder;
      searchInputEl.value = payload.query;
    }
    if (currentQueryEl) currentQueryEl.textContent = payload.query;
    if (clusterNameEl) clusterNameEl.textContent = payload.cluster;
    if (explanationEl) explanationEl.textContent = payload.explanation;
    if (datasetNameEl) datasetNameEl.textContent = key;
    if (nodeTypeEl) nodeTypeEl.textContent = payload.nodeType;
    if (recommendModeEl) recommendModeEl.textContent = payload.recommendMode;
    if (visualTitleEl) visualTitleEl.textContent = payload.visualTitle;

    if (scatterStageEl) {
      scatterStageEl.style.setProperty('--active-x', payload.activePoint.x);
      scatterStageEl.style.setProperty('--active-y', payload.activePoint.y);
    }

    reasonKeyEls.forEach((el, idx) => {
      if (el) el.textContent = payload.reasonKeywords[idx] || '';
    });

    if (metricNmiEl) metricNmiEl.textContent = payload.metrics.nmi;
    if (metricAccEl) metricAccEl.textContent = payload.metrics.acc;
    if (metricAriEl) metricAriEl.textContent = payload.metrics.ari;
    if (metricF1El) metricF1El.textContent = payload.metrics.f1;

    renderExampleChips(payload);
    renderRecommendations(payload.recommendations);

    options.forEach((option) => {
      option.classList.toggle('active', option.dataset.dataset === key);
    });
  }

  if (selectorBtn) {
    selectorBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      setDropdown();
      smoothToShowcase();
    });
  }

  options.forEach((option) => {
    option.addEventListener('click', (event) => {
      event.stopPropagation();
      const chosen = option.dataset.dataset;
      if (!chosen) return;
      renderDataset(chosen);
      setDropdown(false);
      smoothToShowcase();
    });
  });

  if (searchButtonEl && searchInputEl && currentQueryEl) {
    searchButtonEl.addEventListener('click', () => {
      const value = searchInputEl.value.trim();
      if (!value) return;
      currentQueryEl.textContent = value;
    });
  }

  if (searchInputEl && currentQueryEl) {
    searchInputEl.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      const value = searchInputEl.value.trim();
      if (!value) return;
      currentQueryEl.textContent = value;
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

  renderDataset(currentKey);
})();
