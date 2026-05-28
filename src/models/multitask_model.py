"""Multi-task Keras model factory.

Two architectures supplied via the same factory:

* **single_timeframe** (default for CPU smoke and Phase 4): one sequence
  branch + context branch + 4 task heads. Compact and CPU-friendly.
* **mtf_transformer_attention** (full spec): per-timeframe transformer
  encoders with cross-timeframe attention fusion. Inputs include
  fast/main/slow sequences. Use on GPU.

Both share the same output head layout so downstream training code is
agnostic to which architecture was built.

Output heads (in this order):

1. ``direction``      — softmax over 3 classes (down/sideways/up)
2. ``regime``         — softmax over 6 classes
3. ``cycle``          — softmax over 4 classes
4. ``trade_quality``  — sigmoid (binary)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


@dataclass
class ModelConfig:
    seq_len: int
    feature_count_seq: int
    feature_count_context: int = 0
    n_assets: int = 4
    n_timeframes: int = 6
    hidden_size: int = 64
    num_transformer_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.15
    attention_dropout: float = 0.10
    asset_embedding_dim: int = 8
    timeframe_embedding_dim: int = 16
    regime_classes: int = 6
    cycle_classes: int = 4
    use_multi_timeframe_fusion: bool = False
    # When fusion is on:
    fast_seq_len: int = 0
    fast_feature_count: int = 0
    slow_seq_len: int = 0
    slow_feature_count: int = 0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="mindees.multitask")
class _SqueezeAxis(layers.Layer):
    """Serializable replacement for `Lambda(lambda t: tf.squeeze(t, axis=axis))`."""

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.squeeze(x, axis=self.axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg


@keras.saving.register_keras_serializable(package="mindees.multitask")
class _StackOnAxis(layers.Layer):
    """Serializable replacement for `Lambda(lambda t: tf.stack(t, axis=axis))`."""

    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg


@keras.saving.register_keras_serializable(package="mindees.multitask")
class _PositionalEncoding(layers.Layer):
    """Standard sinusoidal positional encoding."""

    def __init__(self, max_len: int, depth: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.depth = depth

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "depth": self.depth})
        return cfg

    def build(self, input_shape):
        position = tf.cast(tf.range(self.max_len), tf.float32)[:, None]
        div_term = tf.exp(
            tf.cast(tf.range(0, self.depth, 2), tf.float32)
            * -(tf.math.log(10000.0) / self.depth)
        )
        angle = position * div_term
        pe_even = tf.sin(angle)
        pe_odd = tf.cos(angle)
        pe = tf.reshape(
            tf.stack([pe_even, pe_odd], axis=-1),
            (self.max_len, -1),
        )[:, : self.depth]
        self.pe = self.add_weight(
            name="pe",
            shape=(self.max_len, self.depth),
            trainable=False,
            initializer=tf.constant_initializer(pe.numpy()),
        )
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len][None, :, :]


def _transformer_block(x, *, num_heads: int, key_dim: int, ff_dim: int,
                       dropout: float, attention_dropout: float, name: str):
    """Pre-norm Transformer encoder block."""
    attn_in = layers.LayerNormalization(name=f"{name}_norm1")(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=attention_dropout,
        name=f"{name}_mha",
    )(attn_in, attn_in)
    x = layers.Add(name=f"{name}_add1")([x, attn_out])
    ffn_in = layers.LayerNormalization(name=f"{name}_norm2")(x)
    ffn_out = layers.Dense(ff_dim, activation="gelu", name=f"{name}_ffn1")(ffn_in)
    ffn_out = layers.Dropout(dropout, name=f"{name}_ffn_dropout")(ffn_out)
    ffn_out = layers.Dense(x.shape[-1], name=f"{name}_ffn2")(ffn_out)
    return layers.Add(name=f"{name}_add2")([x, ffn_out])


def _attention_pool(x, *, name: str):
    """Attention-pooled summary of a sequence."""
    scores = layers.Dense(1, name=f"{name}_score")(x)
    weights = layers.Softmax(axis=1, name=f"{name}_softmax")(scores)
    pooled = layers.Dot(axes=(1, 1), name=f"{name}_dot")([weights, x])
    return _SqueezeAxis(axis=1, name=f"{name}_squeeze")(pooled)


def _sequence_encoder(seq_input, *, cfg: ModelConfig, prefix: str):
    x = layers.LayerNormalization(name=f"{prefix}_input_norm")(seq_input)
    x = layers.Dense(cfg.hidden_size, name=f"{prefix}_proj")(x)
    x = _PositionalEncoding(max_len=seq_input.shape[1], depth=cfg.hidden_size,
                            name=f"{prefix}_pe")(x)
    for i in range(cfg.num_transformer_layers):
        x = _transformer_block(
            x,
            num_heads=cfg.num_heads,
            key_dim=cfg.hidden_size // cfg.num_heads,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            name=f"{prefix}_block{i}",
        )
    return _attention_pool(x, name=f"{prefix}_pool")


def _build_heads(trunk, *, cfg: ModelConfig):
    d = layers.Dense(128, activation="gelu", name="dir_head_d1")(trunk)
    d = layers.Dropout(cfg.dropout, name="dir_head_drop")(d)
    d = layers.Dense(64, activation="gelu", name="dir_head_d2")(d)
    direction = layers.Dense(3, activation="softmax", name="direction", dtype="float32")(d)

    r = layers.Dense(96, activation="gelu", name="reg_head_d1")(trunk)
    r = layers.Dropout(cfg.dropout, name="reg_head_drop")(r)
    r = layers.Dense(48, activation="gelu", name="reg_head_d2")(r)
    regime = layers.Dense(cfg.regime_classes, activation="softmax", name="regime", dtype="float32")(r)

    c = layers.Dense(64, activation="gelu", name="cyc_head_d1")(trunk)
    c = layers.Dropout(cfg.dropout, name="cyc_head_drop")(c)
    c = layers.Dense(32, activation="gelu", name="cyc_head_d2")(c)
    cycle = layers.Dense(cfg.cycle_classes, activation="softmax", name="cycle", dtype="float32")(c)

    t = layers.Dense(64, activation="gelu", name="tq_head_d1")(trunk)
    t = layers.Dropout(cfg.dropout, name="tq_head_drop")(t)
    t = layers.Dense(32, activation="gelu", name="tq_head_d2")(t)
    trade_quality = layers.Dense(1, activation="sigmoid", name="trade_quality", dtype="float32")(t)

    return direction, regime, cycle, trade_quality


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> tf.keras.Model:
    inputs: dict[str, Any] = {}
    inputs["main_sequence"] = layers.Input(
        shape=(cfg.seq_len, cfg.feature_count_seq), name="main_sequence", dtype="float32"
    )
    if cfg.feature_count_context > 0:
        inputs["context"] = layers.Input(
            shape=(cfg.feature_count_context,), name="context", dtype="float32"
        )
    inputs["asset_id"] = layers.Input(shape=(), name="asset_id", dtype="int32")
    inputs["tf_id"] = layers.Input(shape=(), name="tf_id", dtype="int32")

    branches = []

    if cfg.use_multi_timeframe_fusion:
        inputs["fast_sequence"] = layers.Input(
            shape=(cfg.fast_seq_len, cfg.fast_feature_count), name="fast_sequence", dtype="float32"
        )
        inputs["slow_sequence"] = layers.Input(
            shape=(cfg.slow_seq_len, cfg.slow_feature_count), name="slow_sequence", dtype="float32"
        )
        fast_emb = _sequence_encoder(inputs["fast_sequence"], cfg=cfg, prefix="fast")
        slow_emb = _sequence_encoder(inputs["slow_sequence"], cfg=cfg, prefix="slow")
        main_emb = _sequence_encoder(inputs["main_sequence"], cfg=cfg, prefix="main")

        stacked = _StackOnAxis(axis=1, name="fusion_stack")(
            [fast_emb, main_emb, slow_emb]
        )
        fused = layers.MultiHeadAttention(
            num_heads=cfg.num_heads, key_dim=cfg.hidden_size // cfg.num_heads,
            dropout=cfg.attention_dropout, name="cross_tf_attention",
        )(stacked, stacked)
        seq_summary = layers.Flatten(name="fusion_flatten")(fused)
        branches.append(seq_summary)
    else:
        seq_summary = _sequence_encoder(inputs["main_sequence"], cfg=cfg, prefix="main")
        branches.append(seq_summary)

    if cfg.feature_count_context > 0:
        ctx = layers.Dense(128, activation="gelu", name="ctx_d1")(inputs["context"])
        ctx = layers.Dropout(cfg.dropout, name="ctx_drop1")(ctx)
        ctx = layers.Dense(64, activation="gelu", name="ctx_d2")(ctx)
        branches.append(ctx)

    asset_emb = layers.Embedding(
        input_dim=cfg.n_assets, output_dim=cfg.asset_embedding_dim, name="asset_emb",
    )(inputs["asset_id"])
    tf_emb = layers.Embedding(
        input_dim=cfg.n_timeframes, output_dim=cfg.timeframe_embedding_dim, name="tf_emb",
    )(inputs["tf_id"])
    branches.append(asset_emb)
    branches.append(tf_emb)

    trunk = layers.Concatenate(name="trunk_concat")(branches) if len(branches) > 1 else branches[0]
    trunk = layers.Dense(256, activation="gelu",
                         kernel_regularizer=regularizers.l2(1e-5), name="trunk_d1")(trunk)
    trunk = layers.BatchNormalization(name="trunk_bn1")(trunk)
    trunk = layers.Dropout(cfg.dropout, name="trunk_drop1")(trunk)
    trunk = layers.Dense(128, activation="gelu",
                         kernel_regularizer=regularizers.l2(1e-5), name="trunk_d2")(trunk)
    trunk = layers.BatchNormalization(name="trunk_bn2")(trunk)
    trunk = layers.Dropout(cfg.dropout, name="trunk_drop2")(trunk)

    direction, regime, cycle, trade_quality = _build_heads(trunk, cfg=cfg)

    model = models.Model(
        inputs=list(inputs.values()),
        outputs=[direction, regime, cycle, trade_quality],
        name=("mtf_transformer_attention" if cfg.use_multi_timeframe_fusion
              else "single_timeframe_transformer"),
    )
    return model


def encoder_layer_names(model: tf.keras.Model) -> list[str]:
    """Names of layers that constitute the sequence encoder (for freeze/unfreeze)."""
    return [l.name for l in model.layers if l.name.startswith(("main_block", "main_pool", "main_pe",
                                                                 "main_proj", "main_input_norm",
                                                                 "fast_", "slow_"))]
