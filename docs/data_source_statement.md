# 数据来源说明（Data Source Statement）

> 本文件逐一说明仓库中**每一个数据文件**的来源、生成方式与处理流程。

## 0. 一句话总览

| 文件 | 一句话来源与处理流程 |
|---|---|
| `data/raw/annual_reports.csv` | **仿真年报 MD&A 语料**：由 `code/00_generate_demo_data.py` 以固定随机种子（20261007）生成，每篇文本由"中性经营表述句"与"数字化表述句"混合构成，数字化句数与企业潜在数字化水平正相关；真实研究中此处应替换为从巨潮资讯网/CSMAR 批量下载并抽取的"管理层讨论与分析"章节全文 |
| `data/raw/financials.csv` | **仿真公司—年度财务与治理面板**：同由 `00_generate_demo_data.py` 生成，字段口径对齐 CSMAR（规模、杠杆、ROA、成长性、现金持有、董事会特征、SA 指数、周转率、TFP_LP）；真实研究中替换为 CSMAR / Wind 导出 |
| `data/raw/alt_indicators.csv` | **替代指标**：数字技术专利数、研发投入强度、数字类无形资产占比、是否发生数字化并购；真实研究分别取自 CNRDS 专利库、CSMAR 研发与无形资产明细、并购事件库 |
| `data/raw/manual_coding_sample.csv` | **人工编码比对样本**：从面板中随机抽取 200 个公司—年度，按"是否出现数字化战略表述 / 投入承诺 / 落地案例"三项人工打分（1—5 分）并给出四个分维度评分；真实研究中由两名编码者独立编码、计算编码者间一致性（Krippendorff's α） |
| `data/raw/heldout_terms.txt` | **留出词真值表**：语义上同属"企业数字化"、但**未收录进种子词典**的 35 个领域词，用于计算"留出词回收率"这一词典召回效度指标 |
| `data/lexicon/seed_lexicon.txt` | **人工种子词典**：49 个关键词，分"技术应用 / 商业模式 / 战略引领 / 组织赋能"四维，分类框架参考吴非等(2021)、赵宸宇等(2021)、袁淳等(2021) |
| `data/lexicon/segmentation_dict.txt` | **分词专用词典**：保证"数字孪生 / 工业互联网 / 线上线下一体化"等术语切分边界一致；与测度词典相互独立，可包含留出词 |
| `data/lexicon/stopwords_zh.txt` | **中文停用词表**：人工整理，含通用虚词与年报高频无信息词（如"公司""报告期内"） |
| `data/lexicon/expanded_lexicon.csv` | **扩展后测度词典**：`03_build_lexicon.py` 输出。种子词（来源=种子词典）＋ 词向量自动扩展词（来源=词向量扩展），含最大余弦相似度、关联种子词数、语料频次与"是否纳入"标记 |
| `data/lexicon/lexicon_candidates.csv` | **候选扩展词全表**：未通过筛选阈值的候选词也保留，便于人工复核与调整阈值 |
| `data/processed/01_collection_manifest.csv` | **采集清单**：按年度统计文档数、企业数、平均/中位字符数 |
| `data/processed/02_clean_examples.csv` | **清洗前后对照样例**：随机抽取若干篇，展示"清洗前 → 清洗后 → 分词结果"，供人工抽查清洗规则是否误删 |
| `data/processed/03_lexicon_stats.csv` | **词典统计与留出词回收率** |
| `data/processed/firm_year_digital_index.csv` | ⭐ **核心成果：企业—年度数字化转型面板**，含原始词频、对数词频、密度、TF-IDF、四维得分与行业年度去均值后的标准化主指标 |
| `data/processed/panel_regression.csv` | **回归面板**：上述指标与财务/治理/替代指标合并、并按上下 1% 缩尾后的建模数据 |
| `results/tables/T1~T7*.csv/.md` | **结果表**：描述性统计、效度检验、基准回归、机制、异质性、稳健性、内生性 |
| `results/figures/fig1~fig5*.png` | **图表**：指数年度趋势、行业差异与散点、效度散点、四维结构、核心系数图 |

---

## 1. 为什么使用"仿真语料"（诚实披露）

真实的 A 股上市公司年报全文与 CSMAR / CNRDS 财务数据受**版权与登录许可**限制，无法随公开仓库分发。为使"语料采集 → 清洗分词 → 词典构建 → 指标聚合 → 实证建模"这条五步流水线**开箱即可端到端复现**，本仓库以固定随机种子生成一套仿真数据，并保证其**统计性质**贴近真实数据：

- **行业分布**：按证监会 2012 门类设定占比（制造业约 56%、信息技术业约 11%、批发零售业约 9% 等）；
- **规模与杠杆**：总资产对数均值约 22、资产负债率均值约 0.45，均落在 A 股真实区间；
- **文本长度**：每篇 500—1100 字（"MD&A 节选"体量），有效词数约 80—220；
- **测度性质**：数字化词频与"潜在数字化水平"正相关但含泊松型测量误差，从而可在实证中真实地观察到**衰减偏误（attenuation bias）**与**工具变量对衰减的修正**；
- **信息缺失**：语料中 40% 的数字化表述使用"留出词"（种子词典之外），从而制造出真实的词典漏检，使第 3 步的词向量扩展有用武之地。

> ⚠️ **两点必须说明的局限**：
> 1. 为使演示语料在体积可控（约 3 MB）的前提下保留足够的**共现信息**（词向量训练需要共现），仿真语料中数字化表述的密度被**适度放大**（句级占比约 10%—40%，真实年报通常低于 5%）。这会影响指标的量级，但**不影响流水线的可复现性与方法论演示的完整性**。
> 2. 由于数据为仿真生成，本文"实证结果"应被理解为**流水线正确性的验证**（代码能跑通、变量关系能被正确识别、检验能如期工作），而非对真实经济关系的经验发现。真实数据的替换方式见第 3 节。

---

## 2. 五步处理流程与产物对应关系

```
[数据源]  巨潮资讯网年报全文 / CSMAR / CNRDS
   │
   ├─ 01_collect_reports.py     语料采集与入库、字段规范化、完整性体检
   │        └─▶ data/processed/corpus_raw.csv
   │            data/processed/01_collection_manifest.csv
   │
   ├─ 02_clean_and_segment.py   去噪(HTML/URL/页码/纯数字/长英文/标点)
   │                            加载分词词典 → jieba 精确分词 → 去停用词 → 词性过滤
   │        └─▶ data/processed/segmented_corpus.csv
   │            data/processed/02_clean_examples.csv
   │
   ├─ 03_build_lexicon.py       共现 → PPMI → 截断SVD 词向量 → 近邻扩展 → 阈值筛选
   │        └─▶ data/lexicon/expanded_lexicon.csv
   │            data/lexicon/lexicon_candidates.csv
   │            data/processed/03_lexicon_stats.csv
   │
   ├─ 04_compute_index.py       词频 → 分维度 → 长度归一 → TF-IDF → 取对数
   │                            → 行业年度去均值 → 标准化
   │        └─▶ data/processed/firm_year_digital_index.csv   ★核心成果
   │
   └─ 05_regression.py          合并财务数据 → 缩尾 → 描述统计 → 效度检验
                                → 双向固定效应基准回归 → 机制 → 异质性
                                → 稳健性 → IV-2SLS / 滞后 / PSM
            └─▶ data/processed/panel_regression.csv
                results/tables/T1~T7*.{csv,md}
                results/figures/fig1~fig5*.png
```

---

## 3. 如何替换为真实数据

只需替换 `data/raw/` 下的三个输入文件，`01`—`05` 步代码**无需任何修改**：

1. **`data/raw/annual_reports.csv`**：从巨潮资讯网（www.cninfo.com.cn）或 CSMAR"上市公司年报文本"库批量下载 PDF，用 `pdfplumber` / `PyMuPDF` 抽取"管理层讨论与分析"与"董事会报告"章节全文，整理为 `doc_id, stock_code, year, industry, industry_name, title, text` 七列。
2. **`data/raw/financials.csv`**：从 CSMAR 导出财务与治理字段，变量口径见 `docs/variable_dictionary.md`；TFP 可用 `prodest` / `levinsohn-petrin` 包或 CSMAR 已有指标。
3. **`data/raw/alt_indicators.csv`**：从 CNRDS 数字专利库、CSMAR 研发投入与无形资产明细、并购事件库导出。

随后执行：

```bash
pip install -r requirements.txt
python code/run_all.py
```

---

## 4. 数据使用与合规声明

- 本仓库中的 `data/` 目录为**程序生成的仿真数据**，不含任何真实企业信息，可自由使用。
- `data/lexicon/` 中的种子词典与停用词表为人工整理，用于学术研究目的。
- 若使用者替换为真实年报与财务数据，须自行遵守数据提供方的许可协议与相关法律法规，**不得将受限数据二次分发**。
