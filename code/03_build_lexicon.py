# -*- coding: utf-8 -*-
"""
03_build_lexicon.py —— 第 3 步：测度词典构建（种子词典 + 词向量扩展）

构建路径（论文 3.4 节）：
    路径 A（基线）：人工种子词典法（seed-lexicon only）。
    路径 B（主用）：在种子词典基础上，用**词向量语义扩展**自动回收漏检的领域词。
       - 词向量实现：构造词—词共现矩阵 → PPMI 加权 → 截断 SVD 降维（k=100）。
         依据 Levy & Goldberg (2014)：SGNS(Word2Vec 的 Skip-gram + 负采样) 的最优解
         等价于对 PPMI 矩阵做移位分解，因此"PPMI + SVD"是一种无需外部大语料、
         可在小样本年报语料上直接训练的词向量方案；若本机安装了 gensim，
         脚本会额外用 Word2Vec 复核（见文件末尾可选项）。
       - 扩展规则：对每个种子词取 top-K 语义近邻，保留满足
           ① 语料频次 ≥ min_freq_keep
           ② 与种子词最大余弦相似度 ≥ sim_threshold
           ③ 至少与 n_seed_neighbors 个**不同**种子词存在共现
         三个条件的候选词，并按所属最近种子词的维度自动归类。

输入：  data/processed/segmented_corpus.csv
        data/lexicon/seed_lexicon.txt
        data/lexicon/stopwords_zh.txt
        data/raw/heldout_terms.txt（真值，仅用于计算"留出词回收率"）
输出：  data/lexicon/expanded_lexicon.csv   扩展后测度词典（含来源/维度/相似度/是否纳入）
        data/lexicon/lexicon_candidates.csv 全部候选词（供人工复核）
        data/processed/03_lexicon_stats.csv  词典规模与留出词回收率

运行：  python code/03_build_lexicon.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from utils import P_DATA_PROC, P_LEXICON, P_DATA_RAW, set_seed

WINDOW = 10
MIN_FREQ = 5
SVD_DIM = 100
TOP_K_NEIGHBORS = 300
SIM_THRESHOLD = 0.55
MIN_SEED_NEIGHBORS = 3
MIN_FREQ_KEEP = 8
TOPN_EXPAND = 60
PPMI_SHIFT = 1.0


def load_seed_lexicon() -> dict[str, str]:
    path = os.path.join(P_LEXICON, "seed_lexicon.txt")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d, w = line.split("\t")
            out[w] = d
    return out


def load_stopwords() -> set[str]:
    with open(os.path.join(P_LEXICON, "stopwords_zh.txt"), encoding="utf-8") as f:
        return {w.strip() for w in f if w.strip()}


def load_corpus() -> list[list[str]]:
    df = pd.read_csv(os.path.join(P_DATA_PROC, "segmented_corpus.csv"),
                     encoding="utf-8-sig", dtype={"stock_code": str, "doc_id": str})
    return [t.split() for t in df["tokens"].fillna("")]


# ---------------------------------------------------------------- 词向量
def build_word_vectors(docs: list[list[str]]):
    """共现 → PPMI → 截断SVD → 归一化词向量。返回 (vocab, word2idx, vectors, freq)"""
    freq = Counter(w for d in docs for w in d)
    vocab = {w: c for w, c in freq.items() if c >= MIN_FREQ}
    words = sorted(vocab)
    w2i = {w: i for i, w in enumerate(words)}
    n = len(words)

    co = np.zeros((n, n), dtype=np.float64)
    for d in docs:
        ids = [w2i[w] for w in d if w in w2i]
        L = len(ids)
        for i in range(L):
            lo, hi = max(0, i - WINDOW), min(L, i + WINDOW + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                co[ids[i], ids[j]] += 1.0 / abs(i - j)      # 距离加权

    total = co.sum()
    row = co.sum(axis=1, keepdims=True)
    col = co.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co * total) / (row @ col))
    ppmi = np.maximum(pmi - np.log(PPMI_SHIFT), 0.0)
    np.fill_diagonal(ppmi, 0.0)

    k = min(SVD_DIM, min(ppmi.shape) - 1)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    vec = U[:, :k] * np.sqrt(S[:k])
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vec = vec / norms
    return words, w2i, vec, freq


def expand_lexicon(words, w2i, vec, freq, seed: dict[str, str], stopwords: set[str]):
    seed_words = list(seed)
    seed_idx = [w2i[w] for w in seed_words if w in w2i]
    print(f"      种子词 {len(seed_words)} 个，其中 {len(seed_idx)} 个进入词向量词表")

    V = vec
    sim_all = V @ V[seed_idx].T                      # (n_vocab, n_seed)

    records = []
    for i, w in enumerate(words):
        if w in seed:
            continue
        if w in stopwords or len(w) < 2 or w.isdigit():
            continue
        sims = sim_all[i]                              # 长度 = 种子词个数
        order = np.argsort(-sims)
        max_sim = float(sims[order[0]])
        nearest_seed = seed_words[int(order[0])]
        # 与多少个不同种子词的相似度超过阈值（衡量该词是否稳定落在"数字化"语义簇内）
        nb = int(np.sum(sims > 0.20))
        records.append({
            "关键词": w,
            "维度": seed[nearest_seed],
            "最近种子词": nearest_seed,
            "最大相似度": round(max_sim, 4),
            "关联种子词数": nb,
            "语料频次": int(freq[w]),
        })

    cand = pd.DataFrame(records)
    if cand.empty:
        return cand
    cand = cand.sort_values(["最大相似度", "语料频次"], ascending=False).reset_index(drop=True)

    keep_mask = (
        (cand["最大相似度"] >= SIM_THRESHOLD)
        & (cand["语料频次"] >= MIN_FREQ_KEEP)
        & (cand["关联种子词数"] >= MIN_SEED_NEIGHBORS)
    )
    cand["是否纳入"] = np.where(keep_mask, "是", "否")
    # 仅纳入相似度最高的 TOPN 个，防止噪声扩散
    kept_idx = cand.index[keep_mask][:TOPN_EXPAND]
    cand.loc[:, "是否纳入"] = "否"
    cand.loc[kept_idx, "是否纳入"] = "是"
    return cand


def main() -> None:
    set_seed()
    seed = load_seed_lexicon()
    stopwords = load_stopwords()
    docs = load_corpus()
    print(f"[1/4] 语料 {len(docs)} 篇，总词数 {sum(len(d) for d in docs)}")

    words, w2i, vec, freq = build_word_vectors(docs)
    print(f"[2/4] 词表规模 {len(words)}（频次阈值 ≥ {MIN_FREQ}），词向量维度 {vec.shape[1]}")

    cand = expand_lexicon(words, w2i, vec, freq, seed, stopwords)
    print(f"[3/4] 候选扩展词 {len(cand)} 个，其中纳入 {int((cand['是否纳入'] == '是').sum())} 个")

    # ---------------- 组装扩展词典 ----------------
    seed_df = pd.DataFrame({
        "关键词": list(seed), "维度": list(seed.values()), "来源": "种子词典",
        "最近种子词": list(seed), "最大相似度": 1.0, "关联种子词数": len(seed), "语料频次": np.nan,
        "是否纳入": "是",
    })
    if not cand.empty:
        c = cand.copy()
        c["来源"] = "词向量扩展"
        exp_df = pd.concat([seed_df, c[seed_df.columns]], ignore_index=True)
    else:
        exp_df = seed_df
    exp_df.to_csv(os.path.join(P_LEXICON, "expanded_lexicon.csv"), index=False, encoding="utf-8-sig")
    if not cand.empty:
        cand.to_csv(os.path.join(P_LEXICON, "lexicon_candidates.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 留出词回收率（效度证据） ----------------
    ho_path = os.path.join(P_DATA_RAW, "heldout_terms.txt")
    ho = [w.strip() for w in open(ho_path, encoding="utf-8") if w.strip() and not w.startswith("#")]
    added = set(exp_df.loc[(exp_df["来源"] == "词向量扩展") & (exp_df["是否纳入"] == "是"), "关键词"])
    recovered = [w for w in ho if w in added]
    recall = len(recovered) / len(ho) if ho else 0.0

    stats = pd.DataFrame({
        "项目": ["种子词数", "词向量扩展新增词数", "扩展后词典规模",
                "留出词总数", "留出词被回收数", "留出词回收率"],
        "数值": [len(seed), len(added), len(seed) + len(added), len(ho), len(recovered), round(recall, 4)],
    })
    stats.to_csv(os.path.join(P_DATA_PROC, "03_lexicon_stats.csv"), index=False, encoding="utf-8-sig")

    print(f"[4/4] 扩展词典 -> data/lexicon/expanded_lexicon.csv（种子 {len(seed)} + 扩展 {len(added)} = {len(seed) + len(added)} 词）")
    print(f"      留出词回收率 {recall:.1%}（{len(recovered)}/{len(ho)}）")
    if recovered:
        print("      示例回收词：" + "、".join(recovered[:12]))

    # ---------------- 可选：gensim Word2Vec 复核 ----------------
    try:
        from gensim.models import Word2Vec  # noqa
        m = Word2Vec(docs, vector_size=100, window=WINDOW, min_count=MIN_FREQ,
                     sg=1, negative=5, epochs=30, seed=20261007)
        rows = []
        for sw in list(seed)[:10]:
            if sw in m.wv:
                for w, s in m.wv.most_similar(sw, topn=5):
                    rows.append({"种子词": sw, "近邻词": w, "gensim余弦": round(float(s), 4)})
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(P_DATA_PROC, "03_gensim_check.csv"),
                                      index=False, encoding="utf-8-sig")
            print("      [可选] 已输出 gensim Word2Vec 复核表 -> 03_gensim_check.csv")
    except Exception as e:  # gensim 未安装时静默跳过
        print(f"      [可选] 跳过 gensim 复核（{type(e).__name__}）")


if __name__ == "__main__":
    main()
