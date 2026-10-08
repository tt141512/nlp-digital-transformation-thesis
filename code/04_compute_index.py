# -*- coding: utf-8 -*-
"""
04_compute_index.py —— 第 4 步：从文本聚合到"公司—年度"面板指标

计算方式（论文 3.5 节）：
    ① 词频统计：对每篇年报文本，统计扩展词典中各关键词的出现**词次**；
    ② 分维度得分：按"技术应用 / 商业模式 / 战略引领 / 组织赋能"四个维度各自求和；
    ③ 文本长度归一：除词频原值外，另计算**每千词词频密度**（消除文本长度差异）；
    ④ TF-IDF 加权：weight = tf × log(N/df)，降低高频通用词的权重；
    ⑤ 对数化：Dig_ln = ln(1 + 总词频)，缓解右偏；
    ⑥ 行业—年度去均值：Dig = Dig_ln − mean(Dig_ln | 行业×年度)，剔除行业与年度
       共同趋势（如"所有行业都在讲数字化"的时代效应），得到可比的公司—年度指标；
    ⑦ 标准化：对去均值结果做 z-score，使系数具有"标准差变动"的经济含义。

输入：  data/processed/segmented_corpus.csv
        data/lexicon/expanded_lexicon.csv
        data/lexicon/seed_lexicon.txt（用于生成"仅种子词典"的替代指标）
输出：  data/processed/firm_year_digital_index.csv
        data/processed/04_index_summary.csv

运行：  python code/04_compute_index.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from collections import Counter

from utils import P_DATA_PROC, P_LEXICON

DIMS = ["技术应用", "商业模式", "战略引领", "组织赋能"]


def load_lexicon(path: str, only_included: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if only_included and "是否纳入" in df.columns:
        df = df[df["是否纳入"] == "是"]
    return df[["关键词", "维度"]].drop_duplicates(subset=["关键词"]).reset_index(drop=True)


def main() -> None:
    seg = pd.read_csv(os.path.join(P_DATA_PROC, "segmented_corpus.csv"),
                      encoding="utf-8-sig", dtype={"stock_code": str, "doc_id": str})
    lex = load_lexicon(os.path.join(P_LEXICON, "expanded_lexicon.csv"), only_included=True)
    # 仅种子词典（用于生成替代指标 dig_seed_index，做稳健性对照）
    seed_path = os.path.join(P_LEXICON, "seed_lexicon.txt")
    lex_seed = pd.DataFrame(
        [l.rstrip("\n").split("\t")
         for l in open(seed_path, encoding="utf-8")
         if l.strip() and not l.startswith("#")],
        columns=["维度", "关键词"],
    )

    print(f"[1/4] 测度词典 {len(lex)} 词（含扩展）；仅种子词典 {len(lex_seed)} 词")

    kw2dim = dict(zip(lex["关键词"], lex["维度"]))
    seed_kws = set(lex_seed["关键词"])

    # ---------------- 词频统计 ----------------
    rows = []
    doc_tokens = seg["tokens"].fillna("").tolist()
    for i, toks in enumerate(doc_tokens):
        ws = toks.split() if toks else []
        c = Counter(ws)
        hits_total = 0
        hits_seed = 0
        dim_hits = {d: 0 for d in DIMS}
        for w, n in c.items():
            if w in kw2dim:
                hits_total += n
                dim_hits[kw2dim[w]] += n
                if w in seed_kws:
                    hits_seed += n
        rows.append({
            "doc_id": seg["doc_id"].iloc[i],
            "stock_code": seg["stock_code"].iloc[i],
            "year": int(seg["year"].iloc[i]),
            "industry": seg["industry"].iloc[i],
            "industry_name": seg["industry_name"].iloc[i],
            "n_tokens": int(seg["n_tokens"].iloc[i]),
            "dig_hits": hits_total,
            "dig_hits_seed": hits_seed,
            **{f"dim_{d}": dim_hits[d] for d in DIMS},
        })

    df = pd.DataFrame(rows)
    df = df[df["n_tokens"] > 0].reset_index(drop=True)

    # ---------------- TF-IDF ----------------
    N = len(df)
    dfreq = {}
    for toks in doc_tokens:
        for w in set(toks.split()):
            if w in kw2dim:
                dfreq[w] = dfreq.get(w, 0) + 1
    idf = {w: np.log(N / (1 + dfreq.get(w, 0))) for w in kw2dim}

    tfidf_scores = []
    for toks in doc_tokens:
        ws = toks.split()
        c = Counter(w for w in ws if w in kw2dim)
        tfidf_scores.append(float(sum(t * idf[w] for w, t in c.items())))
    df["dig_tfidf"] = tfidf_scores

    # ---------------- 对数化 / 长度归一 / 分维度 ----------------
    df["dig_ln"] = np.log1p(df["dig_hits"])
    df["dig_seed_ln"] = np.log1p(df["dig_hits_seed"])
    df["dig_density"] = df["dig_hits"] / df["n_tokens"] * 1000.0
    for d in DIMS:
        df[f"dim_{d}_ln"] = np.log1p(df[f"dim_{d}"])

    # ---------------- 行业—年度去均值 + 标准化 ----------------
    def demean_z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=0)
        return (s - mu) / (sd if sd > 0 else 1.0)

    grp_key = df["industry"].astype(str) + "_" + df["year"].astype(str)

    def demean_by_group(col: str) -> pd.Series:
        return df[col] - df.groupby(grp_key)[col].transform("mean")

    df["dig_index"] = demean_z(demean_by_group("dig_ln"))                       # 主指标
    df["dig_seed_index"] = demean_z(demean_by_group("dig_seed_ln"))             # 仅种子词典
    df["dig_tfidf_index"] = demean_z(demean_by_group("dig_tfidf"))              # TF-IDF 加权
    df["dig_density_index"] = demean_z(demean_by_group("dig_density"))          # 长度归一
    for d in DIMS:
        df[f"dim_{d}_idx"] = demean_z(demean_by_group(f"dim_{d}_ln"))

    out_cols = ["doc_id", "stock_code", "year", "industry", "industry_name", "n_tokens",
                "dig_hits", "dig_hits_seed", "dig_ln", "dig_density", "dig_tfidf",
                "dig_index", "dig_seed_index", "dig_tfidf_index", "dig_density_index",
                "dig_seed_ln"] + [f"dim_{d}" for d in DIMS] + [f"dim_{d}_idx" for d in DIMS]
    out = df[out_cols].sort_values(["stock_code", "year"]).reset_index(drop=True)
    out.to_csv(os.path.join(P_DATA_PROC, "firm_year_digital_index.csv"), index=False, encoding="utf-8-sig")

    summary = pd.DataFrame({
        "指标": ["词频原值 dig_hits", "对数化 dig_ln", "每千词密度 dig_density",
                "TF-IDF dig_tfidf", "主指标 dig_index"],
        "均值": [df["dig_hits"].mean(), df["dig_ln"].mean(), df["dig_density"].mean(),
                df["dig_tfidf"].mean(), df["dig_index"].mean()],
        "标准差": [df["dig_hits"].std(), df["dig_ln"].std(), df["dig_density"].std(),
                 df["dig_tfidf"].std(), df["dig_index"].std()],
        "最小值": [df["dig_hits"].min(), df["dig_ln"].min(), df["dig_density"].min(),
                 df["dig_tfidf"].min(), df["dig_index"].min()],
        "最大值": [df["dig_hits"].max(), df["dig_ln"].max(), df["dig_density"].max(),
                 df["dig_tfidf"].max(), df["dig_index"].max()],
    }).round(4)
    summary.to_csv(os.path.join(P_DATA_PROC, "04_index_summary.csv"), index=False, encoding="utf-8-sig")

    print(f"[2/4] 公司—年度观测 {len(out)} 个，覆盖 {out['stock_code'].nunique()} 家企业、"
          f"{out['year'].nunique()} 个年度")
    print(f"[3/4] 主指标 dig_index：均值 {df['dig_index'].mean():.3f}，"
          f"标准差 {df['dig_index'].std():.3f}，范围 [{df['dig_index'].min():.2f}, {df['dig_index'].max():.2f}]")
    print(f"[4/4] 已输出 -> data/processed/firm_year_digital_index.csv")


if __name__ == "__main__":
    main()
