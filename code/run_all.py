# -*- coding: utf-8 -*-
"""
run_all.py —— 一键复现整条流水线

用法：
    python run_all.py            # 依次执行 00 → 05
    python run_all.py --build-docx   # 额外把论文正文导出为 Word 文档

五步流水线：
    00  生成演示数据（仿真年报语料 + 公司—年度面板）
    01  语料采集与入库
    02  文本清洗与分词
    03  测度词典构建（种子词典 + 词向量扩展）
    04  指标聚合为公司—年度面板
    05  实证建模 / 效度检验 / 稳健性与内生性处理
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    ("00_generate_demo_data.py", "生成演示数据"),
    ("01_collect_reports.py", "语料采集与入库"),
    ("02_clean_and_segment.py", "文本清洗与分词"),
    ("03_build_lexicon.py", "测度词典构建（种子词典 + 词向量扩展）"),
    ("04_compute_index.py", "指标聚合为公司—年度面板"),
    ("05_regression.py", "实证建模与效度检验"),
]


def main() -> None:
    for script, desc in STEPS:
        print("\n" + "=" * 72)
        print(f"▶ {script}  ——  {desc}")
        print("=" * 72)
        r = subprocess.run([PY, os.path.join(HERE, script)], cwd=HERE)
        if r.returncode != 0:
            raise SystemExit(f"[失败] {script} 退出码 {r.returncode}")

    if "--build-docx" in sys.argv:
        print("\n" + "=" * 72)
        print("▶ build_docx.py  ——  导出论文 Word 文档")
        print("=" * 72)
        subprocess.run([PY, os.path.join(HERE, "build_docx.py")], cwd=HERE, check=False)

    print("\n全部完成。结果见 results/tables 与 results/figures。")


if __name__ == "__main__":
    main()
