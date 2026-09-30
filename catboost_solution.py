import csv
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.parquet"
SAMPLE_PATH = ROOT / "sample_submit.csv"
PREDICTIONS_PATH = ROOT / "catboost_predictions.csv"
HORIZON_DAYS = 30
N_VALIDATION_FOLDS = 3

METRICS = [
    "search", "cat", "searches", "to_cart", "to_ord",
    "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
    "gmv_search", "gmv_cat", "gmv",
]
WINDOWS = [("30d", 29, 0), ("prev30d", 59, 30), ("prev60_90d", 89, 60)]
FEATURES = [f"{metric}_{name}" for name, _, _ in WINDOWS for metric in METRICS]
FEATURES += [f"active_days_{name}" for name, _, _ in WINDOWS]
FEATURES += ["days_since_last"]


def load_user_ids(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"user_id", "predict"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain user_id and predict columns")
        user_ids = [int(row["user_id"]) for row in reader]
    if not user_ids or len(user_ids) != len(set(user_ids)):
        raise ValueError("sample_submit.csv must contain unique user IDs")
    return user_ids


def make_anchor_dates(min_date, max_date):
    first = min_date + timedelta(days=89)
    anchor = max_date - timedelta(days=HORIZON_DAYS)
    dates = []
    while anchor >= first:
        dates.append(anchor)
        anchor -= timedelta(days=HORIZON_DAYS)
    dates.reverse()
    if len(dates) <= N_VALIDATION_FOLDS:
        raise ValueError("Not enough history for training and three non-overlapping validation folds")
    return dates


def extract_features(anchor, user_ids, include_target):
    target_end = anchor + timedelta(days=HORIZON_DAYS) if include_target else anchor
    lf = pl.scan_parquet(TRAIN_PATH).select("event_date", "user_id", *METRICS)
    lf = lf.filter(
        pl.col("event_date").is_between(
            anchor - timedelta(days=89), target_end, closed="both"
        )
    )

    exprs = []
    for name, start_offset, end_offset in WINDOWS:
        start = anchor - timedelta(days=start_offset)
        end = anchor - timedelta(days=end_offset)
        mask = pl.col("event_date").is_between(start, end, closed="both")
        exprs.append(
            pl.when(mask).then(1).otherwise(0).sum().cast(pl.Float32).alias(f"active_days_{name}")
        )
        for metric in METRICS:
            exprs.append(
                pl.when(mask).then(pl.col(metric)).otherwise(0.0)
                .sum().cast(pl.Float32).alias(f"{metric}_{name}")
            )

    last_seen = pl.col("event_date").filter(pl.col("event_date") <= anchor).max()
    exprs.append(
        (pl.lit(anchor, dtype=pl.Date) - last_seen)
        .dt.total_days().fill_null(90).cast(pl.Float32).alias("days_since_last")
    )
    if include_target:
        target_mask = pl.col("event_date").is_between(
            anchor + timedelta(days=1), target_end, closed="both"
        )
        exprs.append(
            pl.when(target_mask).then(pl.col("gmv")).otherwise(0.0)
            .sum().alias("target")
        )

    aggregated = lf.group_by("user_id").agg(exprs).collect(engine="streaming")
    users = pl.DataFrame({"user_id": user_ids, "_order": np.arange(len(user_ids), dtype=np.int32)})
    result = users.join(aggregated, on="user_id", how="left", maintain_order="left").sort("_order")
    result = result.with_columns(pl.col(FEATURES).fill_null(0.0))
    if include_target:
        result = result.with_columns(pl.col("target").fill_null(0.0))
    return result


def rmsle_from_log(y_log, pred_log):
    return float(np.sqrt(np.mean(np.square(y_log - np.maximum(pred_log, 0.0)))))


def make_model(iterations=350):
    return CatBoostRegressor(
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )


def main():
    user_ids = load_user_ids(SAMPLE_PATH)
    dates = (
        pl.scan_parquet(TRAIN_PATH)
        .select(pl.col("event_date").min().alias("min_date"), pl.col("event_date").max().alias("max_date"))
        .collect(engine="streaming")
        .row(0)
    )
    min_date, max_date = dates
    anchors = make_anchor_dates(min_date, max_date)
    validation_anchors = anchors[-N_VALIDATION_FOLDS:]
    print(f"Users: {len(user_ids):,}; train anchors: {len(anchors)}; validation anchors: {validation_anchors}")

    feature_data = {}
    for anchor in anchors:
        print(f"Preparing {anchor} features and target...", flush=True)
        df = extract_features(anchor, user_ids, include_target=True)
        feature_data[anchor] = (
            df.select(FEATURES).to_numpy().astype(np.float32, copy=False),
            df.get_column("target").to_numpy().astype(np.float64, copy=False),
        )
        del df

    fold_scores = []
    baseline_scores = []
    best_iterations = []
    for val_anchor in validation_anchors:
        train_anchors = [a for a in anchors if a + timedelta(days=HORIZON_DAYS) <= val_anchor]
        if not train_anchors:
            raise ValueError(f"No leakage-free training anchors available for {val_anchor}")
        x_train = np.concatenate([feature_data[a][0] for a in train_anchors])
        y_train = np.log1p(np.concatenate([feature_data[a][1] for a in train_anchors])).astype(np.float32)
        x_val, y_val_raw = feature_data[val_anchor]
        y_val = np.log1p(y_val_raw)

        print(f"Training fold {val_anchor} on {len(train_anchors)} anchors...", flush=True)
        model = make_model()
        model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=40)
        pred_log = model.predict(x_val)
        score = rmsle_from_log(y_val, pred_log)
        baseline = x_val[:, FEATURES.index("gmv_30d")]
        baseline_score = rmsle_from_log(y_val, np.log1p(baseline))
        fold_scores.append(score)
        baseline_scores.append(baseline_score)
        best_iterations.append(max(1, model.best_iteration_ + 1))
        print(
            f"  target {val_anchor + timedelta(days=1)}..{val_anchor + timedelta(days=HORIZON_DAYS)}: "
            f"CatBoost RMSLE={score:.6f}; last-30d baseline RMSLE={baseline_score:.6f} "
            f"(best iteration {best_iterations[-1]})"
        )
        del model, x_train, y_train, x_val, y_val, y_val_raw, pred_log

    print(
        f"OOF RMSLE: CatBoost={np.sqrt(np.mean(np.square(fold_scores))):.6f}; "
        f"last-30d baseline={np.sqrt(np.mean(np.square(baseline_scores))):.6f}"
    )
    final_iterations = int(np.median(best_iterations))
    x_train = np.concatenate([feature_data[a][0] for a in anchors])
    y_train = np.log1p(np.concatenate([feature_data[a][1] for a in anchors])).astype(np.float32)
    del feature_data

    print(f"Training final model ({final_iterations} iterations)...", flush=True)
    final_model = make_model(final_iterations)
    final_model.fit(x_train, y_train)
    test_features = extract_features(max_date, user_ids, include_target=False)
    predictions = np.maximum(0.0, np.expm1(final_model.predict(test_features.select(FEATURES).to_numpy())))

    with PREDICTIONS_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "predict"])
        writer.writerows(zip(user_ids, predictions))
    print(f"Saved predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
