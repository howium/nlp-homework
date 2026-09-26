# 作业一 实验报告：文本分类实现

> 课程：自然语言处理 · 作业一：文本分类实现
> 姓名：吴嘉豪 · 学号：2113454
> 数据集：New York Times 新闻三分类（business / politics / sports）

## 1. 实验概述

本实验在 NYT 新闻数据集上，分别使用三类文本表示方法训练分类器，并统一在相同的测试集上比较性能：

1. **Task 1（30 分）词袋模型**：Binary Bag-of-Words 与 Word Frequency 两种稀疏表示 + Logistic Regression；
2. **Task 2（50 分）词向量**：预训练 GloVe（6B, 100d）、自训练 Word2Vec（AG News 语料）、自训练 Word2Vec（NYT 语料），文档向量取词向量平均 + Logistic Regression；
3. **Task 3（20 分）预训练语言模型**：微调 `bert-base-uncased`（max_length=64，3 epochs）。

评价指标为 **Accuracy** 与 **Macro-F1**。由于 NYT 类别分布不均衡（sports 约 75%），Macro-F1 能更公平衡量模型对少数类（business / politics）的识别能力。

**实验环境**：Windows 10，Python 3.12，scikit-learn 1.9 / gensim 4.4 / PyTorch 2.14 (CUDA 12.6) / transformers 5.17，GPU: NVIDIA RTX 3050 Laptop 4GB（BERT 微调使用 fp16）。

## 2. 数据集与预处理

### 2.1 数据集

| 数据集 | 规模 | 用途 |
|---|---|---|
| NYT (`nyt.csv`) | 11,519 篇，text + label 三类：sports 8,639 / politics 1,451 / business 1,429 | 分类主数据集 |
| AG News (`ag.csv`) | 90,000 篇，仅 text（无标签） | 仅用于训练 Word2Vec |

### 2.2 数据划分

对 NYT 全量数据以固定随机种子（`random_state=42`）随机打乱后，按 **80% / 10% / 10%** 划分训练集 / 验证集 / 测试集（分层抽样，保持类别比例一致）：

| 划分 | 样本数 | business | politics | sports |
|---|---|---|---|---|
| Training | 9,215 | 1,143 | 1,161 | 6,911 |
| Validation | 1,152 | 143 | 145 | 864 |
| Test | 1,152 | 143 | 145 | 864 |

划分结果落盘保存（`data/nyt_train/val/test.csv`），**所有实验加载同一份划分**，保证方法间可比。验证集用于超参数选择（如 BERT 学习率），测试集仅用于最终报告。

### 2.3 分词

统一使用 `nltk.word_tokenize`，小写化，仅保留纯字母 token（去除数字与标点，这类 token 无对应词向量且对分类贡献小）。该分词同时用于 BoW 词表构建、Word2Vec 训练与文档向量计算，保证全流程一致。

## 3. Task 1：Bag of Words + Logistic Regression

### 3.1 方法

从**训练集**构建词表（|V| = 57,275）。每篇文档表示为 |V| 维向量：

- **Binary BoW**：词 i 在文档中出现则 x_i = 1，否则为 0（只关心是否出现）；
- **Word Frequency**：x_i 为词 i 在文档中的出现次数。

两种表示均用 scikit-learn `CountVectorizer` 实现（`binary=True/False`），分类器为 `LogisticRegression(max_iter=1000)`。

### 3.2 结果

| 表示方法 | Test Accuracy | Test Macro-F1 |
|---|---|---|
| Binary BoW | 0.9905 | 0.9775 |
| Word Frequency | **0.9913** | **0.9817** |

### 3.3 分析

两种 BoW 表示都达到约 99% 的准确率，说明该任务中"哪些词出现"本身就是很强的分类信号（比如 sports 类的 team、game、season 这类词）。词频版略优于二值版，说明出现次数提供了一定的强度信息（如高频出现的队名、机构名）。但由于新闻文本较短、多数词只出现一次，二者差距很小（0.0008）。

## 4. Task 2：Word Embedding + Logistic Regression

### 4.1 方法

统一使用 100 维词向量，文档向量 = 文档内所有有效单词词向量的**平均**（无词表命中时为零向量，实验中测试集无此类退化样本），分类器同为 Logistic Regression。三组实验：

1. **2a 预训练 GloVe**：官方 `glove.6B.100d.txt`（40 万词，6B 语料训练）；
2. **2b 自训练 Word2Vec（AG News）**：gensim `Word2Vec(vector_size=100, window=5, min_count=5, sg=0, epochs=5)`，在 AG News 全量 90,000 篇上训练；
3. **2c 自训练 Word2Vec（NYT）**：同超参数，仅在 NYT **训练集**（9,215 篇）上训练——严格将验证/测试数据排除在一切训练过程之外。

### 4.2 结果

| 文档表示 | Test Accuracy | Test Macro-F1 |
|---|---|---|
| GloVe 6B 100d（预训练） | 0.9800 | 0.9526 |
| Word2Vec 训练于 AG News | 0.9792 | 0.9500 |
| Word2Vec 训练于 NYT | 0.9809 | 0.9540 |

### 4.3 分析

平均词向量方法整体比 BoW 低约 1 个百分点。平均操作把整篇文档压缩成 100 维，抹平了词序与关键词的尖峰信号；BoW 的 57,275 维稀疏向量则保留了每个判别性词汇的独立权重，而 Logistic Regression 恰好擅长利用这种稀疏线性结构。

W2V-NYT 的训练语料只有 AG News 的约 1/10，分类性能却略高一些。从最近邻词的定性检查看：

| 查询词 | W2V-AG News 最近邻 | W2V-NYT 最近邻 |
|---|---|---|
| president | administration, leader, dushanbe, voltchkov | chairman, secretary, chairwoman, adviser, barack |
| baseball | franchise, owners, nfl, selig | mlb, rivera, rodriguez, yankees |
| economy | economic, recovery, inflation, growth, currency | growth, inflation, economic, market, crisis |

NYT 训练出的向量更贴近该数据集自身的时政与体育词汇，barack、yankees 等正是本任务的强判别词；AG 语料以短财经、体育快讯为主，向量里混进了网球选手这类与 NYT 无关的实体。对平均池化来说，语料的词分布与下游任务是否接近，比语料规模的影响更大。

预训练 GloVe 的表现与 NYT 自训练的 Word2Vec 基本持平（Macro-F1 相差 0.0014）。GloVe 的训练语料大了两个数量级，但并不偏向 NYT 的词汇分布，规模优势没有体现出来，与上面的观察一致。

## 5. Task 3：BERT Fine-tuning

### 5.1 方法

- 模型：`google-bert/bert-base-uncased`（12 层，768 维，1.1 亿参数），`AutoModelForSequenceClassification`，三分类头；
- Tokenization：`max_length=64`，截断，动态 padding（`DataCollatorWithPadding`）；
- 训练：3 epochs，batch size 16，AdamW + 线性 warmup，fp16 混合精度；
- 学习率：按 BERT 论文推荐在验证集上从 {2e-5, 3e-5} 中选择。

### 5.2 结果

| 学习率 | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|---|
| 2e-5 | 0.9792 | 0.9530 | 0.9844 | 0.9657 |
| **3e-5（最终选用）** | **0.9800** | **0.9553** | **0.9844** | **0.9676** |

两个学习率均训练完整 3 个 epoch，按验证集 Macro-F1 选用 3e-5 作为最终模型。

### 5.3 分析

训练损失在 3 个 epoch 内降至约 0.01（接近记住训练集），test 集错误集中在 politics 类（recall 约 0.92–0.96）——该类与 business 共享大量经济政策词汇，是三个类别中最难区分的一对。

微调后 BERT 的 98.44% 准确率高于三种平均词向量方法（97.9%–98.1%），说明上下文相关的表示和端到端微调确实带来了提升：注意力可以聚焦关键短语，而不是被大量次要词的平均稀释。但在该数据规模（训练集仅 9,215 篇）上仍未超过 BoW + LR 基线（99.13%），且学习率 2e-5→3e-5 的变化仅带来 0.002 的 Macro-F1 提升，说明模型已在数据量约束下接近其上限；更大的收益需要更多数据或更长的输入截断长度（64 个 token 对 NYT 长文有信息损失，为作业固定设置）。

## 6. 总体比较与讨论

### 6.1 汇总

| 实验设置 | Test Accuracy | Test Macro-F1 |
|---|---|---|
| Task1: Binary BoW + LR | 0.9905 | 0.9775 |
| Task1: Word Frequency + LR | **0.9913** | **0.9817** |
| Task2a: GloVe 6B 100d（平均）+ LR | 0.9800 | 0.9526 |
| Task2b: Word2Vec（AG News 训练，平均）+ LR | 0.9792 | 0.9500 |
| Task2c: Word2Vec（NYT 训练，平均）+ LR | 0.9809 | 0.9540 |
| Task3: BERT 微调（lr=2e-5） | 0.9844 | 0.9657 |
| Task3: BERT 微调（lr=3e-5，最终） | 0.9844 | 0.9676 |

### 6.2 讨论

效果最好的仍然是最简单的词袋表示。这个任务每类都有大量强特征词，类别之间接近线性可分，BoW 加 Logistic Regression 因此很难被超越；深度模型的优势要有足够的训练数据才能发挥，而这里训练集只有 9,215 篇。

三组词向量实验的结果非常接近，瓶颈应该在平均池化这种表示方式本身：求平均时会丢失词序和关键句的信息，静态向量也处理不了多义词，换不同的训练语料带来的改善有限。

BERT 微调超过了全部三组平均词向量方法，是表现最好的神经方法，但仍没有超过线性基线，这与文献中"强线性基线难以击败"的观察相符。

所有方法的 Accuracy 都比 Macro-F1 高 1 到 3 个百分点，差距来自 sports 占约 75% 的类别不均衡：多数类的正确预测抬高了 Accuracy，而少数类的少量错误在 Macro-F1 中被明显放大，所以评价这类数据时两项指标都要看。

## 7. 结论

1. Word Frequency + Logistic Regression 取得了最好的结果（Accuracy 0.9913 / Macro-F1 0.9817），Binary BoW 紧随其后。对这个线性可分性较好的新闻分类任务，稀疏词袋特征已经足够有效。
2. 三种词向量文档表示性能相近，整体比 BoW 低约 1 个百分点；其中在规模小得多的 NYT 语料上训练的 Word2Vec 反而优于 AG News 语料训练的版本，说明语料与任务的匹配程度比规模更重要。
3. BERT 微调（Accuracy 0.9844 / Macro-F1 0.9676）是最好的神经方法，说明预训练模型对上下文语义的建模有价值，但在本数据规模上没有超过线性基线。
4. Accuracy 与 Macro-F1 的差距来自类别不均衡，少数类的错误对 Macro-F1 影响更大，评价不均衡数据时两项指标都应报告。

## 附录 A：复现说明

见 `hw1/README.md`。在 `hw1/` 目录下运行顺序：

```bash
python src/data_prep.py        # 生成固定数据划分
python src/task1_bow.py        # Task 1
python src/task2_word2vec.py   # Task 2b/2c
python src/task2_glove.py      # Task 2a
python src/task3_bert.py --lr 3e-5   # Task 3（学习率可选）
python src/aggregate_results.py      # 汇总 results/results.csv
```

依赖见 `requirements.txt`；GloVe 下载自 http://nlp.stanford.edu/data/glove.6B.zip （仅需解压出 `glove.6B.100d.txt` 放入 `data/`）；BERT 权重本地缓存于 `models/bert-base-uncased/`。

## 附录 B：文件结构

```
hw1/
├── README.md                  # 运行说明
├── assignment.docx            # 作业要求原文
├── data/
│   ├── raw/                   # nyt.csv、ag.csv（原始数据）
│   └── nyt_train/val/test.csv # 80/10/10 固定划分（脚本生成）
├── src/
│   ├── common.py              # 共享：数据加载 / 分词 / 评估 / 平均池化
│   ├── data_prep.py           # 80/10/10 固定划分
│   ├── task1_bow.py           # Binary BoW / Word Frequency + LR
│   ├── task2_word2vec.py      # Word2Vec (AG / NYT) + LR
│   ├── task2_glove.py         # GloVe 100d + LR
│   ├── task3_bert.py          # BERT 微调（--lr 可调）
│   └── aggregate_results.py   # 结果汇总
├── models/                    # Word2Vec 模型、BERT 权重（不提交）
├── results/                   # results.csv 与 summary.md
└── report/                    # 本报告
```
