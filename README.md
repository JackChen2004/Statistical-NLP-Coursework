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
