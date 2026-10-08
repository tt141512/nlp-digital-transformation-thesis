# -*- coding: utf-8 -*-
"""
00_generate_demo_data.py —— 演示数据生成器（可复现）

⚠️ 重要说明（诚实披露）：
    真实的上市公司年报全文受 CSMAR / 巨潮资讯 / Wind 等数据源的版权与登录限制，
    无法随仓库分发。为了让"语料采集 → 清洗分词 → 词典构建 → 指标聚合 → 实证建模"
    这条五步流水线**开箱即可端到端复现**，本脚本以**固定随机种子**生成一套
    "仿真年报 MD&A 节选语料 + 公司—年度财务面板"，其统计性质（行业分布、规模分布、
    文本长度、数字化词频与真实数字化潜变量的关系）参照 A 股上市公司的公开特征设定。

    真实研究中，只需把本脚本替换为"从数据源导出的年报文本"，后续 01~05 步完全不变。
    详见 docs/data_source_statement.md。

运行：  python code/00_generate_demo_data.py
输出：  data/raw/annual_reports.csv        年报 MD&A 节选语料（doc_id, stock_code, year, ..., text）
        data/raw/financials.csv           公司—年度财务与治理面板
        data/raw/manual_coding_sample.csv 人工编码比对样本（200 份）
        data/raw/alt_indicators.csv       替代指标（是否披露/数字专利/研发强度/无形资产）
        data/lexicon/seed_lexicon.txt     初始种子词典（人工整理）
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from utils import P_DATA_RAW, P_LEXICON, SEED

# =====================================================================
# 一、数字化关键词分类框架（种子词典）
#    分类维度参考吴非等(2021)《企业数字化转型与资本市场表现》的"底层技术运用"与
#    "技术实践应用"两分法，并结合赵宸宇等(2021)、袁淳等(2021)的词表扩展为四维。
# =====================================================================
DIMENSIONS = {
    "技术应用": [
        "人工智能", "大数据", "云计算", "区块链", "物联网", "5G", "数据中心",
        "机器学习", "深度学习", "算法", "数字孪生", "工业互联网", "边缘计算",
        "智能工厂", "数字化车间", "智能装备", "机器人", "自动化",
    ],
    "商业模式": [
        "电子商务", "平台化", "线上线下一体化", "数字化营销", "精准营销",
        "用户画像", "新零售", "跨境电商", "共享经济", "数字营销", "智能客服",
        "直播电商", "私域流量",
    ],
    "战略引领": [
        "数字化转型", "数字化战略", "数字生态", "智能化转型", "数字经济",
        "数字驱动", "战略升级", "数字化升级", "数字中国", "数智化",
    ],
    "组织赋能": [
        "数字化组织", "数字人才", "组织变革", "流程再造", "数字化管理",
        "敏捷组织", "中台", "数字素养", "数字化运营",
    ],
}
KEYWORDS = [k for v in DIMENSIONS.values() for k in v]
KW2DIM = {k: d for d, v in DIMENSIONS.items() for k in v}

# ---------------------------------------------------------------------
# 留出词表（held-out terms）：语义上同属"企业数字化"、但**不在种子词典中**的领域词。
# 它们会被写进语料，用来检验第 3 步"词向量扩展"能否把这些漏网之词自动找回来
# （即测度效度检验中的"留出词回收率"）。真实研究中这批词来自后出文献的新词表。
# ---------------------------------------------------------------------
HELD_OUT_TERMS = [
    "智能制造", "云平台", "智能仓储", "智慧物流", "数字工厂", "智能运维",
    "线上线下融合", "电商平台", "数字化供应链", "智能决策", "数据资产",
    "信息化", "智能化", "网络化", "云服务", "智能终端", "车联网", "智能家居",
    "移动支付", "智慧城市", "智慧工厂", "数据驱动", "平台经济", "智能生产",
    "数字化平台", "云基础设施", "智能识别", "数字技术", "智慧运营", "智能感知",
    "远程运维", "无人车间", "数字化协同", "数字基础设施", "智能供应链",
]

# 通用技术专名（仅用于分词边界，不参与测度）
GENERIC_TECH_TERMS = [
    "供应链", "产业链", "价值链", "数字化", "数据化", "自动化", "信息化", "网络化",
    "智能化", "在线化", "云端", "软件", "系统", "平台", "算法", "模型", "数据",
    "智能", "数字", "互联网", "移动互联网", "信息技术", "工业软件", "操作系统",
    "数据库", "网络安全", "信息安全", "数字技术", "新技术", "技术架构",
]

# =====================================================================
# 二、句式模板（保证关键词可被计数，同时语体接近年报 MD&A）
# =====================================================================
SCENES = ["生产制造", "供应链管理", "市场营销", "客户服务", "研发设计",
          "渠道运营", "内部管理", "质量控制", "物流配送", "风险管理"]

DIGITAL_TEMPLATES = [
    "报告期内，公司以{kw}为抓手，推动{scene}环节的改造升级，相关业务效率稳步提升。",
    "{kw}是公司本轮能力建设的重点方向，期内投入持续加大，已在{scene}形成初步应用。",
    "公司围绕{kw}完成阶段性布局，进一步巩固了{scene}领域的竞争优势。",
    "管理层强调要加快{kw}的推广应用，使其更好服务于{scene}的整体目标。",
    "公司依托{kw}推进{scene}的数字化管理，运营协同能力明显增强。",
    "在{kw}方面，公司持续完善技术架构与人才储备，为{scene}提供支撑。",
    "期内公司加强{kw}建设，{scene}的响应速度与精细化管理水平有所改善。",
    "公司结合{kw}探索新的业务模式，{scene}的覆盖范围进一步扩大。",
]

FILLER = [
    "报告期内，公司坚持以市场为导向，稳步推进各项经营计划，主营业务保持平稳运行。",
    "公司所处行业竞争格局总体稳定，下游需求呈现结构性分化特征。",
    "期内公司持续优化产品结构，加强成本管控，毛利率保持在合理区间。",
    "公司不断完善内部治理机制，强化合规管理与风险防范。",
    "报告期内，公司加大研发投入，产品竞争力得到进一步巩固。",
    "公司积极拓展销售渠道，客户结构持续优化，市场份额稳中有升。",
    "管理层表示，将继续聚焦主业，提升经营质量与盈利能力。",
    "期内原材料价格波动对成本端形成一定压力，公司通过集中采购予以对冲。",
    "公司持续加强安全生产与环保管理，未发生重大安全环保事故。",
    "公司重视投资者关系管理，规范履行信息披露义务。",
    "报告期内公司完成多项技术改造项目，产能利用率有所提升。",
    "公司将进一步优化资源配置，提高资产运营效率。",
    "公司所处的区域市场保持活跃，新增订单情况良好。",
    "期内公司加强应收账款管理，经营性现金流状况改善。",
    "公司积极履行社会责任，参与多项公益捐赠活动。",
    "公司品牌影响力持续提升，获得多项行业荣誉。",
    "管理层认为，未来行业集中度将进一步提升，头部企业优势更加明显。",
    "公司持续完善绩效考核体系，激发员工积极性。",
    "报告期内公司各项业务稳步推进，未出现重大经营风险事项。",
    "公司将继续加强人才队伍建设，为长期发展提供保障。",
    "期内公司完成了生产基地的例行检修，生产计划未受明显影响。",
    "公司对主要供应商进行了重新评估，采购集中度有所提高。",
]

# =====================================================================
# 三、行业设置（证监会 2012 门类）
# =====================================================================
INDUSTRIES = [
    # (代码, 名称, 占比, 数字化基础水平, 数字化增速)
    ("C", "制造业",                 0.56, 0.00, 0.010),
    ("I", "信息传输、软件和信息技术服务业", 0.11, 0.75, 0.022),
    ("F", "批发和零售业",            0.09, 0.25, 0.016),
    ("K", "房地产业",               0.06, -0.10, 0.008),
    ("E", "建筑业",                 0.05, -0.05, 0.010),
    ("G", "交通运输、仓储和邮政业",      0.05, 0.10, 0.013),
    ("D", "电力、热力、燃气及水生产和供应业", 0.05, -0.05, 0.011),
    ("B", "采矿业",                 0.03, -0.20, 0.007),
]

YEARS = list(range(2013, 2024))          # 2013—2023，共 11 年
N_FIRMS = 200
N_DOC_HINT = "仿真 MD&A 节选"


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ---------------- 3.1 企业主表 ----------------
    ind_codes = [c for c, _, _, _, _ in INDUSTRIES]
    ind_p = [p for _, _, p, _, _ in INDUSTRIES]
    ind_base = {c: b for c, _, _, b, _ in INDUSTRIES}
    ind_grow = {c: g for c, _, _, _, g in INDUSTRIES}
    ind_name = {c: n for c, n, _, _, _ in INDUSTRIES}

    firms = pd.DataFrame({
        "stock_code": [f"{600000 + i:06d}" for i in range(N_FIRMS)],
        "industry": rng.choice(ind_codes, size=N_FIRMS, p=ind_p),
    })
    firms["industry_name"] = firms["industry"].map(ind_name)
    firms["soe"] = (rng.random(N_FIRMS) < 0.35).astype(int)
    firms["hightech"] = (
        (firms["industry"] == "I") | ((firms["industry"] == "C") & (rng.random(N_FIRMS) < 0.30))
    ).astype(int)
    firms["ipo_year"] = 2013 - rng.integers(0, 18, size=N_FIRMS)   # 允许 2013 前上市
    firms["board"] = rng.integers(7, 15, size=N_FIRMS)
    firms["indep"] = np.clip(rng.normal(0.375, 0.05, N_FIRMS), 0.3, 0.6)
    firms["dual"] = (rng.random(N_FIRMS) < 0.25).astype(int)
    firms["top1"] = np.clip(rng.normal(0.35, 0.14, N_FIRMS), 0.05, 0.75)
    firms["big4"] = (rng.random(N_FIRMS) < 0.06).astype(int)

    # 企业层面数字化"禀赋"与其个体增速（within 变化来源）；
    # _w_i 为企业对行业—年度数字化披露冲击的暴露系数（工具变量相关性来源）
    firms["_a_i"] = rng.normal(0.0, 0.75, N_FIRMS)
    firms["_g_i"] = rng.normal(0.100, 0.055, N_FIRMS).clip(0.0, 0.38)
    firms["_w_i"] = rng.normal(1.00, 0.40, N_FIRMS).clip(0.1, 2.0)

    panel = firms.merge(pd.DataFrame({"year": YEARS}), how="cross")
    t = panel["year"] - 2013

    # ---------------- 3.2 潜在数字化水平（真实 DGP 的解释变量） ----------------
    latent = (
        panel["_a_i"].values
        + panel["_g_i"].values * t.values
        + panel["industry"].map(ind_base).values
        + panel["industry"].map(ind_grow).values * t.values
        + rng.normal(0, 0.18, len(panel))
    )
    panel["_latent"] = latent
    z = (latent - latent.mean()) / latent.std()
    panel["_z"] = z
    panel["_unit"] = (z - z.min()) / (z.max() - z.min())   # 0~1

    # ---------------- 3.3 财务与治理变量 ----------------
    panel["size"] = 22.0 + 0.09 * t + 0.35 * z + panel["_a_i"] * 0.25 + rng.normal(0, 0.55, len(panel))
    panel["age"] = panel["year"] - panel["ipo_year"]
    panel["roa"] = np.clip(0.045 + 0.012 * z + rng.normal(0, 0.045, len(panel)), -0.30, 0.35)
    panel["lev"] = np.clip(0.45 - 0.025 * z + rng.normal(0, 0.15, len(panel)), 0.03, 0.95)
    panel["growth"] = np.clip(0.12 + 0.035 * z + rng.normal(0, 0.22, len(panel)), -0.60, 1.50)
    panel["cash"] = np.clip(0.18 + 0.020 * z + rng.normal(0, 0.11, len(panel)), 0.01, 0.75)
    panel["board"] = panel["board"] + rng.integers(-1, 2, len(panel))
    panel["indep"] = np.clip(panel["indep"] + rng.normal(0, 0.02, len(panel)), 0.3, 0.6)
    panel["dual"] = panel["dual"]
    panel["top1"] = np.clip(panel["top1"] + rng.normal(0, 0.02, len(panel)), 0.05, 0.75)
    panel["soe"] = panel["soe"]
    panel["hightech"] = panel["hightech"]

    # 机制变量①：融资约束 SA 指数（鞠晓生等, 2013；越接近0约束越强，负值越大约束越弱）
    panel["sa_index"] = -0.737 * panel["size"] + 0.043 * panel["size"] ** 2 - 0.040 * panel["age"] - 0.030 * z
    # 机制变量②：总资产周转率（运营效率）
    panel["turnover"] = np.clip(0.62 + 0.400 * z + rng.normal(0, 0.15, len(panel)), 0.05, 3.0)

    # ---------------- 3.4 被解释变量：全要素生产率 TFP_LP ----------------
    firm_fe = panel.groupby("stock_code")["_a_i"].transform("first").values * 0.25
    year_fe = {y: v for y, v in zip(YEARS, rng.normal(0, 0.10, len(YEARS)))}
    panel["tfp_lp"] = (
        2.30
        + 0.600 * z                                   # ← 待检验的核心效应 β（仿真校准值）
        + 0.210 * (panel["size"] - panel["size"].mean())
        - 0.320 * panel["lev"]
        + 1.100 * panel["roa"]
        + 0.110 * panel["growth"]
        - 0.045 * panel["soe"]
        + firm_fe
        + panel["year"].map(year_fe).values
        + rng.normal(0, 0.16, len(panel))
    )

    # ---------------- 3.5 仿真年报文本 ----------------
    rows = []
    scenes = np.array(SCENES)
    # 行业—年度数字化披露冲击：模拟行业性政策文件/示范项目带动的披露浪潮。
    # 企业按暴露系数 _w_i 差异化受影响（如是否列入试点、是否获得专项补贴），
    # 冲击只影响年报文本表述（披露），不直接进入 TFP 方程 —— 满足工具变量
    # 的排他性约束，为第 5 步"同行业同年度同群均值 IV"提供相关性来源。
    eta = {(ind, int(y)): float(rng.normal(0.0, 2.0))
           for ind in panel["industry"].unique() for y in YEARS}
    w_map = firms.set_index("stock_code")["_w_i"]
    for i, (code, yr, unit, zz) in enumerate(
        zip(panel["stock_code"].values, panel["year"].values, panel["_unit"].values, panel["_z"].values)
    ):
        n_filler = int(rng.integers(12, 20))
        _ind = panel["industry"].values[i]
        lam = max(0.5, 3.0 + 20.0 * unit + float(w_map[code]) * eta[(_ind, int(yr))])
        n_dig = int(rng.poisson(lam))
        if zz > 1.0 and n_dig < 2:              # 高水平企业至少出现 2 句
            n_dig = 2

        sents = list(rng.choice(FILLER, size=n_filler, replace=False))
        for _ in range(n_dig):
            # 25% 的数字化表述使用"留出词"（种子词典之外），以制造真实的漏检
            if rng.random() < 0.25:
                kw = HELD_OUT_TERMS[rng.integers(0, len(HELD_OUT_TERMS))]
            else:
                kw = KEYWORDS[rng.integers(0, len(KEYWORDS))]
            tpl = DIGITAL_TEMPLATES[rng.integers(0, len(DIGITAL_TEMPLATES))]
            sents.append(tpl.format(kw=kw, scene=scenes[rng.integers(0, len(scenes))]))
        rng.shuffle(sents)

        head = f"{yr}年年度报告·管理层讨论与分析（{N_DOC_HINT}）"
        rows.append({
            "doc_id": f"{code}_{yr}",
            "stock_code": code,
            "year": yr,
            "industry": panel["industry"].values[i],
            "industry_name": panel["industry_name"].values[i],
            "title": head,
            "text": "".join(sents),
        })

    reports = pd.DataFrame(rows)

    # ---------------- 3.6 替代指标（用于"替代指标相关性"效度检验） ----------------
    alt = panel[["stock_code", "year", "_unit"]].copy()
    alt["digi_patent"] = rng.poisson(1.2 + 7.0 * alt["_unit"])                 # 数字技术专利数
    alt["rd_ratio"] = np.clip(0.028 + 0.030 * alt["_unit"] + rng.normal(0, 0.012, len(alt)), 0.0, 0.30)
    alt["intangible_digi"] = np.clip(0.05 + 0.06 * alt["_unit"] + rng.normal(0, 0.02, len(alt)), 0.0, 0.45)
    alt["digital_ma"] = (rng.random(len(alt)) < (0.05 + 0.45 * alt["_unit"])).astype(int)  # 数字化并购
    alt = alt.drop(columns=["_unit"])

    # ---------------- 3.7 人工编码比对样本（200 份） ----------------
    sub = panel.sample(n=200, random_state=SEED)[["stock_code", "year", "_unit", "_z"]].copy()
    # 人工评分：依据"是否出现数字化战略表述/投入承诺/落地案例"三项，1~5 分
    human = 1.0 + 3.9 * sub["_unit"] + rng.normal(0, 0.45, len(sub))
    sub["human_score"] = np.clip(human, 1, 5).round(2)
    for d in ["dim_tech", "dim_biz", "dim_strategy", "dim_org"]:
        sub[d] = np.clip(1 + 3.6 * sub["_unit"] + rng.normal(0, 0.6, len(sub)), 1, 5).round(2)
    manual = sub[["stock_code", "year", "human_score", "dim_tech", "dim_biz", "dim_strategy", "dim_org"]]

    # ---------------- 3.8 落盘 ----------------
    fin_cols = ["stock_code", "year", "industry", "industry_name", "soe", "hightech",
                "size", "age", "roa", "lev", "growth", "cash", "board", "indep", "dual",
                "top1", "big4", "sa_index", "turnover", "tfp_lp"]
    financials = panel[fin_cols].copy()

    reports.to_csv(os.path.join(P_DATA_RAW, "annual_reports.csv"), index=False, encoding="utf-8-sig")
    financials.to_csv(os.path.join(P_DATA_RAW, "financials.csv"), index=False, encoding="utf-8-sig")
    alt.to_csv(os.path.join(P_DATA_RAW, "alt_indicators.csv"), index=False, encoding="utf-8-sig")
    manual.to_csv(os.path.join(P_DATA_RAW, "manual_coding_sample.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(P_LEXICON, "seed_lexicon.txt"), "w", encoding="utf-8") as f:
        f.write("# 企业数字化转型种子词典（人工整理，四维分类）\n")
        f.write("# 维度\t关键词\n")
        for d, kws in DIMENSIONS.items():
            for k in kws:
                f.write(f"{d}\t{k}\n")

    # 留出词真值（仅用于"留出词回收率"效度检验，不参与词典构建）
    with open(os.path.join(P_DATA_RAW, "heldout_terms.txt"), "w", encoding="utf-8") as f:
        f.write("# 留出词真值表：语料中出现、但种子词典未收录的数字化领域词\n")
        for w in HELD_OUT_TERMS:
            f.write(w + "\n")

    # 分词专用词典：仅用于保证术语切分边界一致，与"测度词典"相互独立。
    # 它包含种子词、留出词以及通用技术专名，但**不参与第 3 步的测度词典构建**。
    seg_dict = sorted(set(KEYWORDS) | set(HELD_OUT_TERMS) | set(GENERIC_TECH_TERMS))
    with open(os.path.join(P_LEXICON, "segmentation_dict.txt"), "w", encoding="utf-8") as f:
        f.write("# 分词专用词典（保证术语边界一致；不等于测度词典）\n")
        for w in seg_dict:
            f.write(w + "\n")

    print("[OK] annual_reports.csv   ", reports.shape, f"平均文本长度 {int(reports['text'].str.len().mean())} 字")
    print("[OK] financials.csv      ", financials.shape)
    print("[OK] alt_indicators.csv  ", alt.shape)
    print("[OK] manual_coding_sample.csv", manual.shape)
    print(f"[OK] seed_lexicon.txt    种子词 {len(KEYWORDS)} 个 / {len(DIMENSIONS)} 个维度")
    print(f"[OK] heldout_terms.txt   留出词 {len(HELD_OUT_TERMS)} 个")


if __name__ == "__main__":
    main()
