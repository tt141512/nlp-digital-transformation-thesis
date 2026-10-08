# 企业数字化转型的文本测度构建及其对全要素生产率的影响
### —— 基于 A 股上市公司年报文本的 NLP 分析（2013—2023）

> 本仓库是一篇**模拟毕业论文（预演论文）**的完整配套材料：包含论文正文、可运行的五步 NLP 测度流水线代码、企业—年度指标 CSV、结果表与图表、变量字典、数据来源说明与环境依赖。
> 研究问题：**如何用自然语言处理（NLP）方法从年报文本中"测量"企业数字化转型程度，并检验其对企业全要素生产率（TFP）的影响。**
>
> 🌐 已发布在线主页（仓库渲染版）：打开 `index.html` 即可阅读论文全文并下载全部配套文件。本仓库语料与财务面板为固定随机种子生成的仿真数据，用于演示方法链路的可复现性与正确性；真实数据替换方式见 `docs/data_source_statement.md`。

---

## 一、项目简介

数字化转型是"无法直接从财务报表读取"的企业行为，必须借助文本数据构造代理变量。本仓库完整走通一条 NLP 测度构建方法链条：

```
原始年报文本 → 文本清洗与分词 → 测度词典构建（种子词典 + 词向量扩展）
          → 文本聚合为公司—年度面板指标 → 面板计量检验（双向固定效应 / IV-2SLS / PSM）
```

核心方法要点：

| 环节 | 做法 |
|---|---|
| **概念界定** | 数字化划分为「技术应用 / 商业模式 / 战略引领 / 组织赋能」四个维度 |
| **语料来源与跨度** | 沪深 A 股上市公司年报"管理层讨论与分析(MD&A)"章节，2013—2023 年，200 家 × 11 年 = 2,200 个公司—年度 |
| **清洗与分词** | 剔除 HTML/URL/页码/纯数字串/长英文词/标点；**保留 5G、AI 等含字母短代号**；jieba 精确分词 + 领域自定义词典；去停用词 + 词性过滤（仅留名词/动名词/动词/形容词/习用语） |
| **构建路径** | **词典法为主 + 词向量扩展为辅**：共现矩阵 → PPMI 加权 → 截断 SVD（k=100）得到词向量，依据 Levy & Goldberg (2014) 的"SGNS ≈ 移位 PPMI 分解"结论；对种子词取语义近邻并按阈值回收漏检领域词 |
| **文本→面板聚合** | 词频 → 分维度求和 → 每千词密度归一 → TF-IDF 加权 → `ln(1+词频)` → **行业×年度去均值** → 标准化 |
| **效度检验** | ① 与 200 份人工编码评分的 Spearman/Pearson 相关；② 与数字专利、研发强度、数字无形资产的收敛效度；③ 不同测度口径之间的一致性；④ **留出词回收率** |
| **内生性处理** | 工具变量（同行业同年度其他企业数字化均值）2SLS；滞后一期解释变量；倾向得分匹配（PSM）后回归 |

---

## 二、目录结构

```
nlp-dt-thesis/
├── README.md                       本文件：项目简介与复现步骤
├── requirements.txt                环境依赖
├── LICENSE
│
├── paper/
│   ├── 论文正文.md                 ⭐ 论文全文（中英文摘要 / 引言 / 文献综述 / 研究设计 / 实证结果 / 稳健性 / 结论）
│   └── 论文正文.docx               Word 版（由 code/build_docx.py 生成）
│
├── code/                           五步流水线（Python）
│   ├── 00_generate_demo_data.py    生成演示数据（仿真年报语料 + 公司—年度面板）
│   ├── 01_collect_reports.py       【第1步】语料采集与入库
│   ├── 02_clean_and_segment.py     【第2步】文本清洗与分词
│   ├── 03_build_lexicon.py         【第3步】测度词典构建（种子词典 + 词向量扩展）
│   ├── 04_compute_index.py         【第4步】指标聚合为公司—年度面板
│   ├── 05_regression.py            【第5步】实证建模 / 效度检验 / 稳健性 / 内生性
│   ├── build_docx.py               论文正文导出为 Word
│   ├── utils.py                    公共工具（面板双向固定效应、2SLS、缩尾、表格输出）
│   └── run_all.py                  一键复现整条流水线
│
├── data/
│   ├── raw/                        原始输入（年报语料、财务面板、替代指标、人工编码样本、留出词）
│   ├── lexicon/                    种子词典、分词词典、停用词表、扩展词典
│   └── processed/                  ⭐ 中间与最终数据，含 firm_year_digital_index.csv
│
├── results/
│   ├── tables/                     T1 描述性统计 ~ T7 内生性处理（CSV + Markdown）
│   └── figures/                    fig1~fig5 图表
│
└── docs/
    ├── variable_dictionary.md      ⭐ 变量字典（名称、定义、测度、数据来源）
    ├── data_source_statement.md    ⭐ 数据来源说明（每个文件的来源与处理流程）
    └── measurement_construction.md 测度构建说明（概念界定 / 语料 / 清洗规则 / 构建路径 / 聚合公式）
```

---

## 三、快速开始（复现步骤）

```bash
# 1) 准备环境（Python ≥ 3.10）
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) 一键复现全部流程（00 → 05）
python code/run_all.py

# 3) 如需同时导出论文 Word 版
python code/run_all.py --build-docx
```

### 分步运行（便于调试）

```bash
cd code
python 00_generate_demo_data.py    # 生成仿真语料与财务面板
python 01_collect_reports.py       # 语料采集与完整性体检
python 02_clean_and_segment.py     # 清洗 + 分词
python 03_build_lexicon.py         # 词典构建与词向量扩展（含留出词回收率）
python 04_compute_index.py         # 公司—年度数字化转型指标
python 05_regression.py            # 描述统计 / 效度 / 基准 / 机制 / 异质性 / 稳健性 / 内生性
```

> **完全确定性**：所有随机过程均使用固定种子 `20261007`，重复运行结果完全一致。

---

## 四、主要结果表索引

| 文件 | 内容 |
|---|---|
| `results/tables/T1_descriptive` | 表1 主要变量描述性统计 |
| `results/tables/T2_validity` | 表2 测度效度检验（人工编码比对 / 替代指标收敛效度 / 口径一致性 / 留出词回收率） |
| `results/tables/T3_baseline` | 表3 基准回归：数字化转型 → TFP（双向固定效应 + 聚类稳健标准误） |
| `results/tables/T4_mechanism` | 表4 作用机制：融资约束（SA 指数）与运营效率（总资产周转率） |
| `results/tables/T5_heterogeneity` | 表5 异质性：产权性质、高新技术行业、交互项 |
| `results/tables/T6_robustness` | 表6 稳健性：替换测度口径、剔除 2020 年、子样本、更换缩尾、滞后解释变量、替换被解释变量 |
| `results/tables/T7_endogeneity` | 表7 内生性：IV-2SLS、滞后一期、PSM 匹配后回归 |
| `results/figures/fig1~fig5` | 指数年度趋势、行业差异与散点、效度散点图、四维结构、核心系数图 |

---

## 五、环境依赖

见 [`requirements.txt`](requirements.txt)。**刻意保持最小依赖**：

- 不需要 `gensim` / `torch` / `scikit-learn`；
- 词向量由 `numpy` 上的 **PPMI + 截断 SVD** 自行实现（`code/03_build_lexicon.py`）；
- 面板双向固定效应与 2SLS 由 `code/utils.py` 自行实现（迭代式 within 变换 + 聚类稳健标准误）；
- 若本机已装 `gensim`，`03_build_lexicon.py` 会额外输出一份 Word2Vec 复核表（纯可选项，失败自动跳过）。

主要版本：Python 3.10+ / pandas ≥ 2.0 / numpy ≥ 1.24 / scipy ≥ 1.10 / jieba ≥ 0.42.1 / matplotlib ≥ 3.7 / statsmodels ≥ 0.14 / python-docx ≥ 1.1 / tabulate ≥ 0.9

---

## 六、重要说明与局限（请务必阅读）

1. **数据为仿真生成。** 真实年报全文与 CSMAR/CNRDS 财务数据受版权与登录许可限制，无法随仓库分发。本仓库以固定种子生成一套统计性质贴近真实数据的仿真语料与面板，使整条流水线**开箱即可端到端复现**。因此本文实证结果应被理解为**方法链条正确性的验证**，而非对真实经济关系的经验发现。替换为真实数据的做法见 [`docs/data_source_statement.md`](docs/data_source_statement.md)。
2. **仿真语料的数字化词密度被适度放大**（真实年报通常低于 5%），以便在约 3 MB 的体积内保留词向量训练所需的共现信息。这影响指标量级，不影响方法论演示的完整性。
3. 测度效度检验采用了**人工编码比对 + 替代指标相关性 + 多口径一致性 + 留出词回收率**四重证据，是本文相对同类研究在方法透明度上的主要增量。

---

## 七、引用

若参考本仓库的方法框架，可引用：

- 吴非, 胡慧芷, 林慧妍, 任晓怡. 企业数字化转型与资本市场表现——来自股票流动性的经验证据[J]. 管理世界, 2021, 37(7): 130-144.
- 赵宸宇, 王文春, 李雪松. 数字化转型如何影响企业全要素生产率[J]. 财贸经济, 2021, 42(7): 114-129.
- 袁淳, 肖土盛, 耿春晓, 盛誉. 数字化转型与企业分工：专业化还是纵向一体化[J]. 中国工业经济, 2021(9): 137-155.
- Levy O., Goldberg Y. Neural Word Embedding as Implicit Matrix Factorization[C]. NeurIPS, 2014.

---

## 八、许可

代码与文档以 [MIT License](LICENSE) 发布；`data/` 目录为程序生成的仿真数据，可自由使用。
