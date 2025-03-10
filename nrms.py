# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
import tensorflow as tf
import numpy as np


class NRMSModel:
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
    """

    def __init__(
        self,
        hparams: dict,
        word2vec_embedding: np.ndarray = None,
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed: int = None,
    ):
        """Initialization steps for NRMS."""
        # print("NRMS model init")
        self.hparams = hparams
        self.seed = seed

        # SET SEED:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # INIT THE WORD-EMBEDDINGS:
        if word2vec_embedding is None:
            self.word2vec_embedding = np.random.rand(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = word2vec_embedding

        # BUILD AND COMPILE MODEL:
        self.model, self.scorer = self._build_graph()
        data_loss = self._get_loss(self.hparams.loss)
        train_optimizer = self._get_opt(
            optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate
        )
        self.model.compile(loss=data_loss, optimizer=train_optimizer)

    def _get_loss(self, loss: str):
        """Make loss function, consists of data loss and regularization loss
        Returns:
            object: Loss function or loss function name
        """
        if loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError(f"this loss not defined {loss}")
        return data_loss

    def _get_opt(self, optimizer: str, lr: float):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        # TODO: shouldn't be a string input you should just set the optimizer, to avoid stuff like this:
        # => 'WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.'
        if optimizer == "adam":
            train_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")
        return train_opt

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        # print("_build_graph")
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32"
        )
        # print(f"_build_userencoder: his_input_title.shape: {his_input_title.shape}")

        his_input_time = tf.keras.Input(
            shape=(self.hparams.history_size,), dtype="float32"
        )
        # print(f"_build_userencoder: his_input_time.shape: {his_input_time.shape}")

        click_title_presents = tf.keras.layers.TimeDistributed(titleencoder)(
            his_input_title
        )

        # TEST 1: USE NORMALIZED TIME AND SOFTMAX
        # normalized_times = his_input_time / tf.reduce_max(his_input_time)
        # # Add a small value to ensure no weight is zero
        # normalized_times = tf.clip_by_value(normalized_times, clip_value_min=1e-2, clip_value_max=1.0)
        # # Calculate weight based on the time of user clicked news
        # recency_weight = tf.keras.layers.Activation("softmax")(normalized_times)

        # TEST 2: USE RELATIVE TIME AND SOFTMAX
        # Normalize times to relative recency (handle all-zero cases)
        min_time = tf.reduce_min(his_input_time, axis=-1, keepdims=True)
        relative_times = his_input_time - min_time
        print(f"relative_times.shape: {relative_times.shape}")

        # Avoid negative values (in case of missing/invalid data)
        relative_times = tf.maximum(relative_times, 0.0)

        # Scale to seconds and normalize
        scaled_times = relative_times / 1000.0  # Convert ms to seconds
        max_relative = tf.reduce_max(scaled_times, axis=-1, keepdims=True)

        # Handle division by zero when all times are zero (e.g., during validation)
        max_relative = tf.where(max_relative > 0, max_relative, tf.ones_like(max_relative))
        normalized_times = scaled_times / max_relative

        # Clip to avoid extreme values
        clipped_times = tf.clip_by_value(normalized_times, 1e-2, 1.0)

        # Apply softmax
        recency_weight = tf.keras.layers.Activation("softmax")(clipped_times)

        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )

        y = tf.keras.layers.Multiply()([y, tf.expand_dims(recency_weight, axis=-1)])

        user_present = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)

        model = tf.keras.Model([his_input_title, his_input_time], user_present, name="user_encoder")
        return model

    def _build_newsencoder(self):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        embedding_layer = tf.keras.layers.Embedding(
            self.word2vec_embedding.shape[0],
            self.word2vec_embedding.shape[1],
            weights=[self.word2vec_embedding],
            trainable=True,
        )
        sequences_input_title = tf.keras.Input(
            shape=(self.hparams.title_size,), dtype="int32"
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = tf.keras.layers.Dropout(self.hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [y, y, y]
        )
        y = tf.keras.layers.Dropout(self.hparams.dropout)(y)
        pred_title = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)

        model = tf.keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        # print("Building NRMS model")
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size),
            dtype="int32",
        )
        # print(f"_build_nrms: his_input_title.shape: {his_input_title.shape}")
        his_input_time = tf.keras.Input(
            shape=(self.hparams.history_size, ), dtype="float32"
        )
        # print(f"_build_nrms: his_input_time.shape: {his_input_time.shape}")
        pred_input_title = tf.keras.Input(
            # shape = (hparams.npratio + 1, hparams.title_size)
            shape=(None, self.hparams.title_size),
            dtype="int32",
        )
        pred_input_title_one = tf.keras.Input(
            shape=(
                1,
                self.hparams.title_size,
            ),
            dtype="int32",
        )
        pred_title_one_reshape = tf.keras.layers.Reshape((self.hparams.title_size,))(
            pred_input_title_one
        )
        titleencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        user_present = self.userencoder([his_input_title, his_input_time])
        news_present = tf.keras.layers.TimeDistributed(self.newsencoder)(
            pred_input_title
        )
        news_present_one = self.newsencoder(pred_title_one_reshape)

        preds = tf.keras.layers.Dot(axes=-1)([news_present, user_present])
        preds = tf.keras.layers.Activation(activation="softmax")(preds)

        pred_one = tf.keras.layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)

        model = tf.keras.Model([his_input_title, his_input_time, pred_input_title], preds)
        scorer = tf.keras.Model([his_input_title, his_input_time, pred_input_title_one], pred_one)

        return model, scorer
