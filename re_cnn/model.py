# re_cnn/model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_re_cnn(
    emb_matrix: np.ndarray,
    max_len: int = 256,
    max_dist: int = 64,
    pos_dim: int = 32,
    filters: int = 128,
    kernel_sizes=(3,4,5),
    dropout: float = 0.3,
    num_labels: int = 2,
    trainable_word_emb: bool = False,
):
    vocab_size, emb_dim = emb_matrix.shape

    w_in = keras.Input(shape=(max_len,), dtype="int32", name="word_ids")
    p1_in = keras.Input(shape=(max_len,), dtype="int32", name="pos1_ids")
    p2_in = keras.Input(shape=(max_len,), dtype="int32", name="pos2_ids")

    word_emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        weights=[emb_matrix],
        trainable=trainable_word_emb,
        mask_zero=False,
        name="word_emb",
    )(w_in)

    pos_vocab = 2 * max_dist + 1
    pos1_emb = layers.Embedding(pos_vocab, pos_dim, name="pos1_emb")(p1_in)
    pos2_emb = layers.Embedding(pos_vocab, pos_dim, name="pos2_emb")(p2_in)

    x = layers.Concatenate(name="concat_emb")([word_emb, pos1_emb, pos2_emb])
    x = layers.Dropout(dropout)(x)

    pooled = []
    for k in kernel_sizes:
        c = layers.Conv1D(filters=filters, kernel_size=k, padding="same", activation="relu")(x)
        p = layers.GlobalMaxPooling1D()(c)
        pooled.append(p)

    x = layers.Concatenate()(pooled) if len(pooled) > 1 else pooled[0]
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_labels, activation="softmax")(x)

    model = keras.Model(inputs=[w_in, p1_in, p2_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=2e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
