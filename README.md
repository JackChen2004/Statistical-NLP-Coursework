# Statistical-NLP-Coursework
CNN Branch
NER Reference Repo: https://github.com/niccolot/NER_biLSTM-CNN?tab=readme-ov-file
Embedding: https://github.com/ncbi-nlp/BioWordVec?tab=readme-ov-file



🧬 CNN based BC5CDR NER–RE Pipeline

Two-stage biomedical information extraction pipeline:

PubTator → NER (Chemical/Disease) → RE (CID) → Document-level CID prediction

Supports:
	•	Independent NER evaluation
	•	Independent RE evaluation (gold entities)
	•	Full end-to-end pipeline evaluation (predicted entities)

⸻

📂 Project Structure

Statistical-NLP-Coursework/
│
├── data/
│   ├── origin/                    # Raw BC5CDR PubTator files
│   └── processed/                 # Sentence-level CoNLL + sentence index
│
├── embeddings/
│   └── biowordvec_bc5_subset.txt  # 200d biomedical embeddings
│
├── NER_biLSTM-CNN-master/         # NER model (BiLSTM + CNN-char)
│
├── PrepareREData.py               # Build RE gold training data
├── build_sentence_index.py        # Align sentence-level NER back to PMID
├── RE_Test_Pred.py                # Full NER→RE pipeline inference
│
└── re_data_gold/                  # RE gold dataset (auto-generated)


⸻

1️⃣ NER Module

Model
	•	BiLSTM + CNN-char
	•	Word embeddings: BioWordVec (200d)

Training Data

data/processed/bc5cdr_ner_sentence_{train,dev,test}.txt

Sentence-level CoNLL format:

token   X   X   B-CHEM
token   X   X   O
(blank line = sentence boundary)

Train

Inside NER_biLSTM-CNN-master/:

python train.py

Model saved to:

models/bilstm_cnn_bc5_sentence.keras


⸻

2️⃣ Embeddings

Using domain-specific biomedical embeddings:

embeddings/biowordvec_bc5_subset.txt

	•	200-dimensional
	•	Subset extracted from BioWordVec
	•	Initialized as non-trainable

⸻

3️⃣ RE Gold Data Preparation

Script:

python PrepareREData.py

Input:

data/origin/CDR_*Set.PubTator.txt

Output:

re_data_gold/
  re_train_gold.tsv
  re_dev_gold.tsv
  re_test_gold.tsv

Format:

LABEL \t sentence_with_<e1>_<e2>_markers

Example:

CID    <e1> aspirin </e1> reduces <e2> headache </e2>

Used to train RE model with gold entities.

⸻

4️⃣ Sentence Index (NER–RE Bridge)

Script:

python build_sentence_index.py

Output:

data/processed/
  bc5cdr_sentence_index_{train,dev,test}.jsonl

Each record:

{
  "pmid": "...",
  "sent_id": 3,
  "tokens": [...],
  "token_offsets": [...]
}

Purpose:
	•	Align sentence-level NER output back to document level
	•	Required for end-to-end pipeline

⸻

5️⃣ End-to-End Pipeline

Script:

python RE_Test_Pred.py

Steps:
	1.	Load trained NER model
	2.	Predict BIO tags on test set
	3.	Convert predicted entities back to document-level
	4.	Build RE test instances using predicted entities

Output:

pipeline_outputs/
  test_pred_entities.jsonl
  re_test_pred.tsv

This is the predicted-entity RE test set.

⸻

🔁 Replacing the RE Model

NER is frozen.
You can plug in any RE model.

Training

Use:

re_data_gold/re_train_gold.tsv
re_data_gold/re_dev_gold.tsv

If your model requires:
	•	Different format
	•	Entity positions
	•	Token indices

Use:

re_data_gold/re_train_gold.jsonl

which contains:
	•	token list
	•	chem_span
	•	dis_span
	•	mesh ids
	•	pmid
	•	sentence id

⸻

Testing (Pipeline Evaluation)

Use:

pipeline_outputs/re_test_pred.tsv

Predict labels on this file to evaluate full pipeline performance.

⸻

📊 Evaluation Settings

We support three evaluation modes:

Setting	Description
NER only	BIO F1
RE (gold entities)	Upper bound
RE (predicted entities)	End-to-end pipeline


⸻

🧠 Design Principles

✔ Modular

NER and RE are fully decoupled.

✔ Replaceable RE

Any new RE model only needs to adapt to:

re_train_gold.tsv
re_test_pred.tsv

✔ Error Propagation Analysis

Compare:
	•	RE + gold entities
	•	RE + predicted entities

to analyze NER impact.

⸻

🚀 Quick Start (Full Pipeline)

# 1. Train NER
cd NER_biLSTM-CNN-master
python train.py

# 2. Prepare RE gold data
cd ..
python PrepareREData.py

# 3. Build sentence index
python build_sentence_index.py

# 4. Train RE baseline (your model)

# 5. Run pipeline inference
python RE_Test_Pred.py


⸻

📌 Current Status

✔ NER trained
✔ Biomedical embeddings integrated
✔ RE gold dataset prepared
✔ Sentence alignment implemented
✔ End-to-end inference script ready

Next step: experiment with different RE models.

