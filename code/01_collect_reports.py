# -*- coding: utf-8 -*-
"""
01_collect_reports.py —— 第 1 步：语料采集与入库

职责：
    把数据源导出的年报文本统一为规范语料表，并做完整性体检，输出采集清单(manifest)。
    真实研究中本步对应"从 CSMAR / 巨潮资讯网 / Wind 批量下载年报 PDF 并抽取
    '管理层讨论与分析(MD&A)'与'董事会报告'章节文本"。

输入：  data/raw/annual_reports.csv
输出：  data/processed/corpus_raw.csv         规范化语料
        data/processed/01_collection_manifest.csv  采集清单（按年份统计）

运行：  python code/01_collect_reports.py
"""
from __future__ import annotations

import os
import pandas as pd

from utils import P_DATA_RAW, P_DATA_PROC

REQUIRED = ["doc_id", "stock_code", "year", "industry", "title", "text"]
YEAR_MIN, YEAR_MAX = 2013, 2023


def main() -> None:
    src = os.path.join(P_DATA_RAW, "annual_reports.csv")
    if not os.path.exists(src):
        raise SystemExit(
            "未找到 data/raw/annual_reports.csv，请先运行  python code/00_generate_demo_data.py"
        )

    df = pd.read_csv(src, encoding="utf-8-sig", dtype={"stock_code": str})
    print(f"[1/4] 读入原始语料 {df.shape[0]} 条")

    # ---- 1. 字段完整性检查 ----
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise SystemExit(f"缺少必要字段: {miss}")

    # ---- 2. 清洗异常记录 ----
    n0 = len(df)
    df = df.dropna(subset=["text", "stock_code", "year"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 50]                       # 过短文本剔除
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]
    df = df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    print(f"[2/4] 剔除空值/过短/越界/重复：{n0 - len(df)} 条，保留 {len(df)} 条")

    # ---- 3. 语料规范字段 ----
    df["char_len"] = df["text"].str.len()
    df["doc_id"] = df["doc_id"].astype(str)

    # ---- 4. 采集清单（按年度 / 行业统计覆盖面） ----
    manifest = (
        df.groupby("year")
        .agg(文档数=("doc_id", "count"),
             企业数=("stock_code", "nunique"),
             平均字符数=("char_len", "mean"),
             中位字符数=("char_len", "median"))
        .reset_index()
    )
    manifest["平均字符数"] = manifest["平均字符数"].round(0).astype(int)
    manifest["中位字符数"] = manifest["中位字符数"].round(0).astype(int)

    ind = (
        df.groupby(["industry", "industry_name"])
        .agg(企业数=("stock_code", "nunique"), 文档数=("doc_id", "count"))
        .reset_index()
    )

    df.to_csv(os.path.join(P_DATA_PROC, "corpus_raw.csv"), index=False, encoding="utf-8-sig")
    manifest.to_csv(os.path.join(P_DATA_PROC, "01_collection_manifest.csv"), index=False, encoding="utf-8-sig")
    ind.to_csv(os.path.join(P_DATA_PROC, "01_collection_by_industry.csv"), index=False, encoding="utf-8-sig")

    print("[3/4] 语料规范化完成 -> data/processed/corpus_raw.csv")
    print(f"[4/4] 面板覆盖：{df['stock_code'].nunique()} 家企业 × {df['year'].nunique()} 年 "
          f"= {df['stock_code'].nunique() * df['year'].nunique()} 个潜在公司—年度，实际 {len(df)} 篇文本")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
