# -*- coding: utf-8 -*-
"""
02_clean_and_segment.py —— 第 2 步：文本清洗与分词

处理规则（论文 3.3 节所描述的"文本清洗与分词规则"的代码实现）：
    (1) 结构化噪声剔除：HTML 标签、URL、邮箱、页码页眉("第X页/共X页")、空白与换行；
    (2) 数值剔除：剔除纯数字串，但**保留含字母的短代号**（如 5G、AI、O2O），
        以免误删领域词；
    (3) 英文剔除：剔除长度≥3 的纯英文单词；保留中文、字母数字混合的短代号；
    (4) 标点与特殊符号剔除，仅保留中文字符、字母与数字；
    (5) 分词：jieba 精确模式，并**加载领域自定义词典**（种子词典 + 常见数字技术专名），
        解决"数字孪生/工业互联网/线上线下一体化"等词被错误切分的问题；
    (6) 去停用词：加载 data/lexicon/stopwords_zh.txt；
    (7) 词性过滤：仅保留名词/动名词/动词/形容词/习用语/简称等实词词性，
        过滤"的/了/在"等虚词与单字虚词。

输入：  data/processed/corpus_raw.csv
        data/lexicon/seed_lexicon.txt（用于构建领域词典）
        data/lexicon/stopwords_zh.txt
输出：  data/processed/segmented_corpus.csv   分词结果（doc_id, tokens 空格分隔, n_tokens）
        data/processed/02_clean_examples.csv 清洗前后对照样例（便于人工抽查）

运行：  python code/02_clean_and_segment.py
"""
from __future__ import annotations

import os
import re
import pandas as pd
import jieba
import jieba.posseg as pseg

from utils import P_DATA_PROC, P_LEXICON

# ---------------------------------------------------------------- 清洗正则
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
RE_EMAIL = re.compile(r"\S+@\S+")
RE_PAGENO = re.compile(r"第\s*\d+\s*页(?:\s*/?\s*共\s*\d+\s*页)?|共\s*\d+\s*页")
RE_PUREDIGIT = re.compile(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])")   # 纯数字串（保留 5G）
RE_ENGWORD = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{3,}(?![A-Za-z0-9])")  # 长英文词
RE_KEEP = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9]")                  # 非中英文数字一律删

# ---------------------------------------------------------------- 词性白名单
KEEP_POS = {
    "n", "nr", "ns", "nt", "nz", "nl", "ng",          # 名词类
    "vn", "v", "vd",                                   # 动词/动名词
    "a", "an", "ag", "al",                             # 形容词
    "j", "l", "i",                                     # 简称/习用语/成语
    "eng",                                             # 英文与字母数字代号(5G/AI)
}

# 领域自定义词典由 data/lexicon/segmentation_dict.txt 提供（见 00_generate_demo_data.py）
SEG_DICT_FILE = "segmentation_dict.txt"


def load_seg_dict() -> list[str]:
    """加载分词专用词典（仅保证术语边界一致，不等于"测度词典"）。"""
    path = os.path.join(P_LEXICON, SEG_DICT_FILE)
    words = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith("#"):
                    words.append(w)
    return words


def load_stopwords() -> set[str]:
    path = os.path.join(P_LEXICON, "stopwords_zh.txt")
    with open(path, encoding="utf-8") as f:
        return {w.strip() for w in f if w.strip()}


def clean_text(text: str) -> str:
    s = str(text)
    s = RE_HTML.sub("", s)
    s = RE_URL.sub("", s)
    s = RE_EMAIL.sub("", s)
    s = RE_PAGENO.sub("", s)
    s = RE_PUREDIGIT.sub("", s)
    s = RE_ENGWORD.sub("", s)
    s = RE_KEEP.sub("", s)
    return s


def segment(text: str, stopwords: set[str]) -> list[str]:
    out = []
    for w, pos in pseg.cut(text):
        w = w.strip()
        if not w or len(w) < 2:                 # 去掉单字词（多为虚词/噪声）
            continue
        if w in stopwords:
            continue
        if pos not in KEEP_POS:
            continue
        if w.isdigit():
            continue
        out.append(w)
    return out


def main() -> None:
    src = os.path.join(P_DATA_PROC, "corpus_raw.csv")
    if not os.path.exists(src):
        raise SystemExit("请先运行  python code/01_collect_reports.py")
    df = pd.read_csv(src, encoding="utf-8-sig", dtype={"stock_code": str, "doc_id": str})

    stopwords = load_stopwords()
    domain = load_seg_dict()
    for w in domain:
        # 关键：必须显式指定词性标签。jieba.add_word 的 tag 默认为 'x'，
        # 会被下面的词性白名单过滤掉，导致领域词全部丢失。
        jieba.add_word(w, freq=20000, tag="nz")
    print(f"[1/3] 停用词 {len(stopwords)} 个；分词自定义词典 {len(domain)} 个词已加载")

    cleaned = df["text"].map(clean_text)
    tokens = [segment(t, stopwords) for t in cleaned]
    print("[2/3] 清洗 + 分词完成")

    res = df[["doc_id", "stock_code", "year", "industry", "industry_name"]].copy()
    res["n_tokens"] = [len(t) for t in tokens]
    res["tokens"] = [" ".join(t) for t in tokens]
    res = res[res["n_tokens"] > 0].reset_index(drop=True)
    res.to_csv(os.path.join(P_DATA_PROC, "segmented_corpus.csv"), index=False, encoding="utf-8-sig")

    # 清洗前后对照样例（供人工抽查）
    ex = pd.DataFrame({
        "doc_id": df["doc_id"].head(5),
        "清洗前": df["text"].head(5).str.slice(0, 60),
        "清洗后": cleaned.head(5).str.slice(0, 60),
        "分词结果": ["/".join(t[:25]) for t in tokens[:5]],
    })
    ex.to_csv(os.path.join(P_DATA_PROC, "02_clean_examples.csv"), index=False, encoding="utf-8-sig")

    print(f"[3/3] 分词语料 -> data/processed/segmented_corpus.csv")
    print(f"      平均有效词数 {res['n_tokens'].mean():.1f} 词/篇；"
          f"最短 {res['n_tokens'].min()}，最长 {res['n_tokens'].max()}")


if __name__ == "__main__":
    main()
