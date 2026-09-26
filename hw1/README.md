# HW1: Text Classification on NYT（文本分类实现）

对 NYT 新闻数据集（business / politics / sports 三分类）分别使用
**词袋模型、Word Embedding / Word2Vec、预训练语言模型（BERT）** 三类文本表示
训练分类器，统一在相同的 80/10/10 划分的 Test Set 上比较 Accuracy 与 Macro-F1。

**实验报告（最终交付物）**：[report/hw1_report.pdf](report/hw1_report.pdf)

## 环境

使用仓库根目录的共享虚拟环境（见根目录 README），HW1 依赖见根目录
`requirements.txt`。NLTK 分词数据需一次性下载：

```python
import nltk; nltk.download("punkt"); nltk.download("punkt_tab")
```

## 数据

- `data/raw/nyt.csv`：NYT 新闻，`text` + `label`（主分类数据集，11,519 篇）
- `data/raw/ag.csv`：AG News 文本（仅用于训练 Word2Vec，90,000 篇）
- `src/data_prep.py`：固定 `random_state=42` 随机打乱后按 80/10/10
  **分层划分** train/val/test，落盘到 `data/nyt_{train,val,test}.csv`
  （生成文件不提交）。**所有实验加载同一份划分**，保证可比性。

## 运行（在 hw1/ 目录下按顺序执行）

```bash
python src/data_prep.py        # 生成固定数据划分
python src/task1_bow.py        # Task 1: Binary BoW / Word Frequency + LR
python src/task2_word2vec.py   # Task 2b/2c: Word2Vec(AG) / Word2Vec(NYT) + LR
python src/task2_glove.py      # Task 2a: GloVe 6B 100d + LR
python src/task3_bert.py --lr 3e-5   # Task 3: BERT fine-tuning（学习率可选）
python src/aggregate_results.py      # 汇总 results/results.csv
```

## 外部资源（一次性，均不提交）

- GloVe：下载 http://nlp.stanford.edu/data/glove.6B.zip ，解压出
  `glove.6B.100d.txt` 放到 `data/` 下。
- BERT 权重：`google-bert/bert-base-uncased`，本地缓存于
  `models/bert-base-uncased/`（国内网络可设置
  `HF_ENDPOINT=https://hf-mirror.com` 或从 ModelScope 下载；
  已校验权重 SHA256 与官方一致）。

## 结果

所有实验的 Accuracy / Macro-F1 汇总在 `results/results.csv`，
完整指标与分析见 `report/hw1_report.pdf`。
