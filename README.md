BioREDirect 论文复现项目

本仓库为 UCL Statistical NLP 课程项目 —— BioREDirect 模型复现。

我们复现的论文为：

Lu et al., 2023
BioREDirect: Direct Biomedical Relation Extraction with Soft Prompts

本项目目标：

✅ 复现论文在 BioRED (BC8) 数据集上的结果

✅ 复现论文在 CDR 数据集上的结果

✅ 复现预训练模型效果

✅ 从 BioLinkBERT 重新训练模型

✅ 对比论文报告指标与复现结果

项目结构
.
├── data/                  # 数据集（不上传到 git）
├── models/                # 预训练模型权重（不上传）
├── outputs/               # 实验输出结果（不上传）
├── scripts/               # 复现流程脚本
│   ├── 00_env_setup_colab.sh
│   ├── 01_download_data_and_models_colab.sh
│   ├── 02_eval_pretrained_bc8_colab.sh
│   ├── 03_train_bc8_from_biolinkbert_colab.sh
│   └── 04_predict_pubtator_to_pubtator_colab.sh
├── src/                   # 原始 BioREDirect 源代码
├── requirements.txt
└── README.md
说明：

大文件（数据、模型、输出）通过 .gitignore 忽略

代码仓库只包含可复现实验流程

环境配置

推荐环境：

Python 3.11

CUDA GPU

PyTorch (CUDA 版本)

安装依赖：
pip install -r requirements.txt
若使用 Colab：
bash scripts/00_env_setup_colab.sh


数据集

本项目使用两个数据集：

1️⃣ BioRED (BC8)

论文主要实验数据集，用于生物医学关系抽取。

2️⃣ CDR

Chemical-Disease Relation 数据集。

下载数据和预训练模型：

bash scripts/01_download_data_and_models_colab.sh


下载后数据位于：

data/

实验一：评估预训练 BioREDirect（BC8）

复现论文中报告的 BC8 测试结果：

bash scripts/02_eval_pretrained_bc8_colab.sh


核心运行命令：

python src/run_exp.py \
    --in_bioredirect_model bioredirect_biored_pt \
    --in_test_tsv_file datasets/bioredirect/processed/bc8_test.tsv \
    --num_epochs 0


论文报告（BC8 Strict ALL）F1 ≈ 0.49
我们的复现结果：≈ 0.49

说明：该实验为加载官方权重直接评估。


实验二：从 BioLinkBERT 重新训练（BC8）

从 BioLinkBERT 初始化训练 BioREDirect：

bash scripts/03_train_bc8_from_biolinkbert_colab.sh


主要超参数：

soft_prompt_len = 8

learning_rate = 1e-5

batch_size = 16

max_seq_len = 512

num_epochs = 10

训练完成后模型保存在：

outputs/


实验三：PubTator 格式预测

流程：

PubTator → TSV

模型预测

TSV → PubTator

运行：

bash scripts/04_predict_pubtator_to_pubtator_colab.sh

评估指标

评估指标包括：

Precision

Recall

F1-score

分为：

Strict

Relaxed

ALL

评估实现位于：

src/evaluation.py
