from tensorflow.keras.backend import clear_session
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import numpy as np
import gc
import os

#######################
#       CONFIG        #
#######################
USE_TIMESTAMPS = False
SUBSAMPLE_DATASET = True
TEST = False
PATH = Path("/dtu/blackhole/01/203937/DeepLearning/")
#######################
#######################

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_IS_BEYOND_ACCURACY_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
)

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import (
    slice_join_dataframes,
    concat_str_columns,
    chunk_dataframe,
    split_df,
)
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec import NRMSModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    if USE_TIMESTAMPS:
        df_history = (
            pl.scan_parquet(path.joinpath("history.parquet"))
            .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL)
            .pipe(
                truncate_history,
                column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                history_size=history_size,
                padding_value=0,
                enable_warning=False,
            )
            .pipe(
                truncate_history,
                column=DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
                history_size=history_size,
                padding_value=0,
                enable_warning=False,
            )
        )
    else:
        df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
            )
        )

    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    return df_behaviors


DUMP_DIR = Path("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
SEED = np.random.randint(0, 1_000)

MODEL_NAME = f"NRMS-{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}-{SEED}"

MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")
TEST_DF_DUMP = DUMP_DIR.joinpath("test_predictions", MODEL_NAME)
TEST_DF_DUMP.mkdir(parents=True, exist_ok=True)

print(f"Dir: {MODEL_NAME}")

DATASPLIT = "ebnerd_small"

#########################
# HYPERTUNED PARAMETERS #
#########################
MAX_TITLE_LENGTH = 30 
HISTORY_SIZE = 25
EPOCHS = 20
#########################
#########################

FRACTION = 1.0
FRACTION_TEST = 1.0
#
hparams_nrms.history_size = HISTORY_SIZE

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 32
BATCH_SIZE_TEST_WO_B = 32
BATCH_SIZE_TEST_W_B = 4
N_CHUNKS_TEST = 10
CHUNKS_DONE = 0

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]

if USE_TIMESTAMPS:
    COLUMNS.append(DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL)


df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .sample(fraction=FRACTION)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=SEED,
    )
    .pipe(create_binary_labels_column)
)

if SUBSAMPLE_DATASET:
    df_train = df_train[:1000]
    df_train, df_validation = split_df(df_train, fraction=0.9, seed=SEED, shuffle=False)
else:
    df_validation = (
        ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE)
        .select(COLUMNS)
        .pipe(create_binary_labels_column)
        .sample(fraction=FRACTION)
    )

df_articles = pl.read_parquet(PATH.joinpath(DATASPLIT, "articles.parquet"))

print(f"Lenght of df_train: {len(df_train)}")
print(f"Lenght of df_validation: {len(df_validation)}")

###############################################################
# BEST PERFORMING MODEL: distilbert-base-multilingual-cased   #
###############################################################
TRANSFORMER_MODEL_NAME = "distilbert-base-multilingual-cased"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)

# =>
print("Init train- and val-dataloader")
if USE_TIMESTAMPS:
    train_dataloader = NRMSDataLoader(
        behaviors=df_train,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        timestamp_column=DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
        eval_mode=False,
        batch_size=BATCH_SIZE_TRAIN,
    )
    val_dataloader = NRMSDataLoader(
        behaviors=df_validation,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        timestamp_column=DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
        eval_mode=True,
        batch_size=BATCH_SIZE_VAL,
    )
else:
    train_dataloader = NRMSDataLoaderPretransform(
        behaviors=df_train,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False,
        batch_size=BATCH_SIZE_TRAIN,
    )
    val_dataloader = NRMSDataLoaderPretransform(
        behaviors=df_validation,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=BATCH_SIZE_VAL,
    )

print("Init model")
model = NRMSModel(
    hparams=hparams_nrms,
    word2vec_embedding=word2vec_embedding,
    seed=42,
)
print("Model initialized")

print("Fitting model")
hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=EPOCHS,
)
print("Model fitted")

print(f"saving model: {MODEL_WEIGHTS}")
model.model.save_weights(MODEL_WEIGHTS)
# print(f"loading model: {MODEL_WEIGHTS}")
# model.model.load_weights(MODEL_WEIGHTS)

pred_validation = model.scorer.predict(val_dataloader)
df_validation = add_prediction_scores(df_validation, pred_validation.tolist()).pipe(
    add_known_user_column, known_users=df_train[DEFAULT_USER_COL]
)
metrics = MetricEvaluator(
    labels=df_validation["labels"].to_list(),
    predictions=df_validation["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
print(metrics.evaluate())

del (
    transformer_tokenizer,
    transformer_model,
    train_dataloader,
    val_dataloader,
    df_validation,
    df_train,
)
gc.collect()

if TEST:
    # =>
    print("Init df_test")
    df_test = (
        ebnerd_from_path(PATH.joinpath("ebnerd_testset", "test"), history_size=HISTORY_SIZE)
        .sample(fraction=FRACTION_TEST)
        .with_columns(
            pl.col(DEFAULT_INVIEW_ARTICLES_COL)
            .list.first()
            .alias(DEFAULT_CLICKED_ARTICLES_COL)
        )
        .select(COLUMNS + [DEFAULT_IS_BEYOND_ACCURACY_COL])
        .with_columns(
            pl.col(DEFAULT_INVIEW_ARTICLES_COL)
            .list.eval(pl.element() * 0)
            .alias(DEFAULT_LABELS_COL)
        )
    )

    # Split test in beyond-accuracy. BA samples have more 'article_ids_inview'.
    df_test_wo_beyond = df_test.filter(~pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))
    df_test_w_beyond = df_test.filter(pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))

    df_test_chunks = chunk_dataframe(df_test_wo_beyond, n_chunks=N_CHUNKS_TEST)
    df_pred_test_wo_beyond = []

    for i, df_test_chunk in enumerate(df_test_chunks[CHUNKS_DONE:], start=1 + CHUNKS_DONE):
        print(f"Init test-dataloader: {i}/{len(df_test_chunks)}")
        # Initialize DataLoader
        if USE_TIMESTAMPS:
            test_dataloader_wo_b = NRMSDataLoader(
                behaviors=df_test_chunk,
                article_dict=article_mapping,
                unknown_representation="zeros",
                history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                timestamp_column=DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
                eval_mode=True,
                batch_size=BATCH_SIZE_TEST_WO_B,
            )
        else:
            test_dataloader_wo_b = NRMSDataLoaderPretransform(
                behaviors=df_test_chunk,
                article_dict=article_mapping,
                unknown_representation="zeros",
                history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                eval_mode=True,
                batch_size=BATCH_SIZE_TEST_WO_B,
            )
        print(f"DataLoader initialized for chunk {i}")

        # Predict and clear session
        scores = model.scorer.predict(test_dataloader_wo_b)
        print(f"Predictions done for chunk {i}")
        clear_session()
        print(f"Session cleared for chunk {i}")

        # Process the predictions
        df_test_chunk = add_prediction_scores(df_test_chunk, scores.tolist()).with_columns(
            pl.col("scores")
            .map_elements(lambda x: list(rank_predictions_by_score(x)))
            .alias("ranked_scores")
        )
        print(f"Predictions processed for chunk {i}")

        # Save the processed chunk
        df_test_chunk.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
            TEST_DF_DUMP.joinpath(f"pred_wo_ba_{i}.parquet")
        )
        print(f"Chunk {i} saved")

        # Append and clean up
        df_pred_test_wo_beyond.append(df_test_chunk)
        print(f"Chunk {i} appended")

        # Cleanup
        del df_test_chunk, test_dataloader_wo_b, scores
        gc.collect()
        print(f"Cleanup done for chunk {i}")

    # =>
    df_pred_test_wo_beyond = pl.concat(df_pred_test_wo_beyond)
    df_pred_test_wo_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        TEST_DF_DUMP.joinpath("pred_wo_ba.parquet")
    )

    print("Init test-dataloader: beyond-accuracy")
    if USE_TIMESTAMPS:
        test_dataloader_w_b = NRMSDataLoader(
            behaviors=df_test_w_beyond,
            article_dict=article_mapping,
            unknown_representation="zeros",
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            timestamp_column=DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
            eval_mode=True,
            batch_size=BATCH_SIZE_TEST_W_B,
        )
    else:
        test_dataloader_w_b = NRMSDataLoaderPretransform(
            behaviors=df_test_w_beyond,
            article_dict=article_mapping,
            unknown_representation="zeros",
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            eval_mode=True,
            batch_size=BATCH_SIZE_TEST_W_B,
        )

    scores = model.scorer.predict(test_dataloader_w_b)
    df_pred_test_w_beyond = add_prediction_scores(
        df_test_w_beyond, scores.tolist()
    ).with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("ranked_scores")
    )
    df_pred_test_w_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        TEST_DF_DUMP.joinpath("pred_w_ba.parquet")
    )

    # # =>
    df_test = pl.concat([df_pred_test_wo_beyond, df_pred_test_w_beyond])
    df_test.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        TEST_DF_DUMP.joinpath("pred_concat.parquet")
    )

    write_submission_file(
        impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
        prediction_scores=df_test["ranked_scores"],
        path=DUMP_DIR.joinpath("predictions.txt"),
        filename_zip=f"{DATASPLIT}_predictions-{MODEL_NAME}.zip",
    )
