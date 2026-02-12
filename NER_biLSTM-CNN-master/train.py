import preprocessing
import architecture
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


lr = 1e-3
epochs = 50
batch_size = 64

# === Data paths (BC5 CDR sentence-level CoNLL) ===
TRAIN_PATH = '../data/processed/bc5cdr_ner_sentence_train.txt'
DEV_PATH   = '../data/processed/bc5cdr_ner_sentence_dev.txt'
EMB_PATH   = '../embeddings/biowordvec_bc5_subset.txt'

train_data_list, val_data_list, labels_train, labels_val = preprocessing.get_dataset(
    TRAIN_PATH,
    DEV_PATH,
    EMB_PATH
)

word2idx, \
    case2idx, \
    char2idx, \
    label2idx, \
    word_embeddings, \
    case_embeddings = preprocessing.get_dicts_and_embeddings(
        TRAIN_PATH,
        DEV_PATH,
        EMB_PATH
    )

# Keep these consistent with preprocessing padding/truncation.
# For sentence-level BC5, the repo defaults are typically OK.
MAX_LEN_SEQ = 50
MAX_LEN_CHARS = 15

model = architecture.build_model(
    word2idx=word2idx,
    case2idx=case2idx,
    char2idx=char2idx,
    label2idx=label2idx,
    word_embeddings=word_embeddings,
    case_embeddings=case_embeddings,
    max_len_seq=MAX_LEN_SEQ,
    max_len_chars=MAX_LEN_CHARS,
)

opt = keras.optimizers.Adam(learning_rate=lr)
loss = keras.losses.SparseCategoricalCrossentropy()

model.compile(loss=loss, optimizer=opt)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)]

model.summary()

# =========================
# Padding mask as sample_weight (ignore PADDING tokens in loss)
# =========================
pad_id = label2idx.get("PADDING", 0)
train_sample_weight = (labels_train != pad_id).astype("float32")
val_sample_weight = (labels_val != pad_id).astype("float32")

history = model.fit(x=train_data_list,
                    y=labels_train,
                    sample_weight=train_sample_weight,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(val_data_list, labels_val, val_sample_weight),
                    callbacks=callbacks)

# =========================
# 1) Save model
# =========================
os.makedirs("models", exist_ok=True)
model.save("models/bilstm_cnn_bc5_sentence.keras")
print("Model saved to models/bilstm_cnn_bc5_sentence.keras")

# =========================
# 2) Predict on validation set
# =========================
import numpy as np

pred_probs = model.predict(val_data_list)
pred_labels = np.argmax(pred_probs, axis=-1)

# =========================
# 3) Simple token-level accuracy
# =========================
true_labels = labels_val
pad_id = label2idx.get("PADDING", 0)
mask = true_labels != pad_id
correct = (pred_labels == true_labels) & mask
accuracy = correct.sum() / mask.sum()

print(f"Token-level accuracy (val): {accuracy:.4f}")

# =========================
# 3b) Entity-level (span) evaluation with seqeval
# =========================
# Install: pip install seqeval
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

pad_id = label2idx.get("PADDING", -1)
idx2label = {v: k for k, v in label2idx.items()}

# Convert ids -> label strings and strip padding for seqeval
true_seqs = []
pred_seqs = []

for t_seq, p_seq in zip(true_labels, pred_labels):
    if pad_id != -1:
        keep = t_seq != pad_id
        t_seq = t_seq[keep]
        p_seq = p_seq[keep]
    true_seqs.append([idx2label[int(i)] for i in t_seq])
    pred_seqs.append([idx2label[int(i)] for i in p_seq])

p = precision_score(true_seqs, pred_seqs)
r = recall_score(true_seqs, pred_seqs)
f1 = f1_score(true_seqs, pred_seqs)

print("\n=== Entity-level (seqeval) on VAL ===")
print(f"Precision: {p:.4f}")
print(f"Recall:    {r:.4f}")
print(f"F1:        {f1:.4f}\n")
print("=== Detailed report ===")
print(classification_report(true_seqs, pred_seqs, digits=4))

# =========================
# 4) Export predictions for RE pipeline
# =========================

os.makedirs("predictions", exist_ok=True)
output_path = "predictions/val_ner_predictions.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for sent_preds in pred_labels:
        labels_str = [idx2label[idx] for idx in sent_preds]
        f.write(" ".join(labels_str) + "\n")

print(f"NER predictions saved to {output_path}")
