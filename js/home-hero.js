(function () {
  const root = document.getElementById("home-hero-root");

  if (!root || !window.React || !window.ReactDOM) {
    return;
  }

  const e = React.createElement;

  const insights = [
    ["01", "Attribute-aware", "融合节点属性与高阶拓扑结构", "merge"],
    ["02", "Intra-edge Contrast", "在超边内部学习稳定一致性", "contrast"],
    ["03", "End-to-End", "聚类目标与表征学习联合优化", "flow"],
  ];

  const nodes = [
    "left-[16%] top-[26%]",
    "left-[34%] top-[16%]",
    "left-[58%] top-[25%]",
    "left-[78%] top-[18%]",
    "left-[24%] top-[64%]",
    "left-[49%] top-[72%]",
    "left-[73%] top-[60%]",
  ];

  function HeroButton({ href, label, icon, primary }) {
    const className = primary
      ? "hero-primary-button group inline-flex min-h-[3.5rem] items-center justify-center gap-3 rounded-full px-8 text-sm font-bold uppercase tracking-[0.14em] transition duration-300 hover:-translate-y-1 focus:outline-none"
      : "group inline-flex min-h-[3.5rem] items-center justify-center gap-3 rounded-full border border-teal-200 bg-white/78 px-8 text-sm font-bold uppercase tracking-[0.14em] text-slate-800 shadow-[0_18px_36px_rgba(20,184,166,0.12)] backdrop-blur-sm transition duration-300 hover:-translate-y-1 hover:border-teal-400 hover:text-teal-700 focus:outline-none focus:ring-4 focus:ring-teal-100";

    return e(
      "a",
      { className, href },
      e("span", null, label),
      e("i", {
        className:
          icon +
          " text-[0.8rem] transition duration-300 group-hover:translate-x-0.5",
        "aria-hidden": "true",
      }),
    );
  }

  function LinearIcon({ type }) {
    const common = {
      fill: "none",
      stroke: "currentColor",
      strokeWidth: "1.7",
      strokeLinecap: "round",
      strokeLinejoin: "round",
    };

    const icons = {
      merge: [
        e("circle", Object.assign({ key: "m1", cx: "6", cy: "7", r: "2.2" }, common)),
        e("circle", Object.assign({ key: "m2", cx: "6", cy: "17", r: "2.2" }, common)),
        e("circle", Object.assign({ key: "m3", cx: "18", cy: "12", r: "2.6" }, common)),
        e("path", Object.assign({ key: "m4", d: "M8 8.1l7.5 3" }, common)),
        e("path", Object.assign({ key: "m5", d: "M8 15.9l7.5-3" }, common)),
      ],
      contrast: [
        e("circle", Object.assign({ key: "c1", cx: "8", cy: "12", r: "4" }, common)),
        e("circle", Object.assign({ key: "c2", cx: "16", cy: "12", r: "4" }, common)),
        e("path", Object.assign({ key: "c3", d: "M12 8v8" }, common)),
        e("path", Object.assign({ key: "c4", d: "M5 5l2 2" }, common)),
        e("path", Object.assign({ key: "c5", d: "M19 19l-2-2" }, common)),
      ],
      flow: [
        e("path", Object.assign({ key: "f1", d: "M4 7h8a4 4 0 010 8H8" }, common)),
        e("path", Object.assign({ key: "f2", d: "M9 11l-4 4 4 4" }, common)),
        e("path", Object.assign({ key: "f3", d: "M15 5l5 5-5 5" }, common)),
      ],
    };

    return e(
      "svg",
      {
        className: "h-5 w-5 text-teal-600",
        viewBox: "0 0 24 24",
        "aria-hidden": "true",
      },
      icons[type],
    );
  }

  function InsightItem({ index, title, text, icon }) {
    return e(
      "li",
      {
        className:
          "grid grid-cols-[2.75rem_1fr] gap-4 border-t border-slate-200/80 py-4 text-left first:border-t-0",
      },
      e(
        "span",
        {
          className:
            "flex h-10 w-10 items-center justify-center rounded-full border border-teal-200 bg-white/80 text-teal-600 shadow-sm",
        },
        e(LinearIcon, { type: icon }),
      ),
      e(
        "span",
        null,
        e("span", { className: "font-mono text-xs font-bold leading-6 text-teal-600" }, index),
        e("strong", { className: "block text-sm font-bold text-slate-950" }, title),
        e("span", { className: "mt-1 block text-sm leading-6 text-slate-600" }, text),
      ),
    );
  }

  function HypergraphIllustration() {
    return e(
      "aside",
      {
        className:
          "hero-visual relative hidden min-h-[28rem] perspective-[1200px] lg:block",
        "aria-label": "Hypergraph contrastive learning illustration",
      },
      e("div", {
        className:
          "hero-plane absolute inset-x-6 top-10 h-[18rem] rounded-lg border border-slate-200/80 bg-white/62 shadow-[0_30px_80px_rgba(15,23,42,0.10)] backdrop-blur-md",
      }),
      e("div", {
        className:
          "hero-plane hero-plane-back absolute inset-x-14 top-3 h-[18rem] rounded-lg border border-teal-100/90 bg-teal-50/35",
      }),
      e("div", {
        className:
          "absolute left-[14%] top-[24%] h-[14rem] w-[20rem] rotate-[-10deg] rounded-[48%] border border-blue-200/90",
      }),
      e("div", {
        className:
          "absolute left-[28%] top-[18%] h-[16rem] w-[18rem] rotate-[12deg] rounded-[44%] border border-teal-200/90",
      }),
      e("div", {
        className:
          "absolute left-[24%] top-[38%] h-[9.5rem] w-[23rem] rotate-[-2deg] rounded-[48%] border border-slate-300/80",
      }),
      e("span", {
        className:
          "hero-orbit hero-orbit-one absolute left-[17%] top-[34%] h-px w-[62%] origin-left rotate-[-7deg] rounded-full",
        "aria-hidden": "true",
      }),
      e("span", {
        className:
          "hero-orbit hero-orbit-two absolute left-[28%] top-[47%] h-px w-[52%] origin-left rotate-[5deg] rounded-full",
        "aria-hidden": "true",
      }),
      e("span", {
        className:
          "hero-orbit hero-orbit-three absolute left-[22%] top-[56%] h-px w-[58%] origin-left rotate-[-2deg] rounded-full",
        "aria-hidden": "true",
      }),
      nodes.map((position, index) =>
        e(
          "span",
          {
            key: position,
            className:
              position +
              " hero-node absolute flex h-4 w-4 items-center justify-center rounded-full bg-slate-950 shadow-[0_0_0_8px_rgba(20,184,166,0.12)]",
            style: { animationDelay: `${index * 0.22}s` },
          },
          e("span", {
            className: "h-1.5 w-1.5 rounded-full bg-white",
            "aria-hidden": "true",
          }),
        ),
      ),
      e(
        "div",
        {
          className:
            "absolute bottom-0 left-8 right-8 rounded-lg border border-slate-200/80 bg-white/86 px-6 py-5 text-left shadow-[0_18px_50px_rgba(15,23,42,0.08)] backdrop-blur-md",
        },
        e(
          "div",
          {
            className:
              "text-xs font-bold uppercase tracking-[0.24em] text-teal-600",
          },
          "Minimal Research Visual",
        ),
        e(
          "div",
          { className: "mt-2 text-xl font-black text-slate-950" },
          "Hyperedge-level Representation",
        ),
        e(
          "p",
          { className: "mt-2 mb-0 text-sm leading-6 text-slate-600" },
          "以细线、层叠平面和轻微运动表达高阶关系，保持阅读区域干净清晰。",
        ),
      ),
    );
  }

  function HomeHero() {
    return e(
      "div",
      {
        className:
          "relative min-h-screen overflow-hidden bg-white bg-cover bg-center",
        style: { backgroundImage: 'url("assets/img/bg-hypergraph-light.png")' },
      },
      e("div", {
        className:
          "absolute inset-0 bg-[linear-gradient(90deg,rgba(255,255,255,0.985)_0%,rgba(255,255,255,0.94)_48%,rgba(255,255,255,0.78)_100%)]",
      }),
      e("div", {
        className: "hero-grid-texture absolute inset-0 opacity-[0.16]",
        "aria-hidden": "true",
      }),
      e("div", {
        className:
          "absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-white to-transparent",
      }),
      e(
        "div",
        {
          className:
            "relative mx-auto grid min-h-screen max-w-7xl items-center gap-16 px-5 pb-16 pt-36 sm:px-8 lg:grid-cols-[1.08fr_0.92fr] lg:px-10 lg:pb-20 lg:pt-40",
        },
        e(
          "section",
          { className: "max-w-4xl text-left" },
          e(
            "div",
            {
              className:
                "mb-8 flex flex-col gap-3 sm:flex-row sm:items-center",
            },
            e(
              "span",
              {
                className:
                  "text-xs font-black uppercase tracking-[0.34em] text-blue-700",
              },
              "IntraHC / Attributed Hypergraph Clustering",
            ),
            e("span", { className: "hidden h-px w-12 bg-teal-400 sm:block" }),
            e(
              "span",
              {
                className:
                  "text-xs font-bold uppercase tracking-[0.22em] text-teal-600",
              },
              "Contrastive Signal",
            ),
          ),
          e(
            "h1",
            {
              className:
                "m-0 max-w-5xl text-[3rem] font-black leading-[1.02] tracking-normal text-slate-950 sm:text-[4.35rem] lg:text-[5.7rem]",
            },
            "让高阶关系聚类更稳健",
          ),
          e(
            "p",
            {
              className:
                "mt-8 max-w-2xl text-lg leading-9 text-slate-600 sm:text-xl",
            },
            "IntraHC 通过超边内部对比学习，将节点属性、拓扑结构与聚类目标统一到端到端框架中，提升复杂属性超图场景下的表征质量，并在多个基准数据集上",
            e(
              "strong",
              {
                className:
                  "mx-1 inline-block text-2xl font-black text-teal-700 sm:text-3xl",
              },
              "实现 SOTA 性能",
            ),
            "。",
          ),
          e(
            "div",
            { className: "mt-10 flex flex-col gap-4 sm:flex-row" },
            e(HeroButton, {
              href: "demo.html",
              label: "立即试用",
              icon: "fas fa-arrow-right",
              primary: true,
            }),
            e(HeroButton, {
              href: "algorithm.html",
              label: "了解详情",
              icon: "fas fa-file-lines",
              primary: false,
            }),
          ),
          e(
            "ul",
            {
              className:
                "mt-12 grid max-w-3xl list-none gap-0 p-0 sm:grid-cols-3 sm:gap-6",
            },
            insights.map(([index, title, text, icon]) =>
              e(InsightItem, { key: index, index, title, text, icon }),
            ),
          ),
        ),
        e(HypergraphIllustration),
      ),
    );
  }

  ReactDOM.createRoot(root).render(e(HomeHero));
})();
