from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


NUMERIC_COLS = [
    "text_length",
    "text_density",
    "link_count",
    "link_text_ratio",
    "section_count",
    "div_count",
    "li_count",
    "h1_count",
    "h2_count",
    "h3_count",
    "heading_total",
    "js_ui_score",
    "empty_body",
]

CATEGORICAL_COLS = [
    "segmentation_type",
]


LOG1P_COLS = [
    "text_length",
    "link_count",
    "section_count",
    "div_count",
    "li_count",
]


def build_preprocessor() -> ColumnTransformer:
    def log1p_selected(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in LOG1P_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(log1p_selected, validate=False)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])

    return preprocessor


def fit_kmeans(df: pd.DataFrame, n_clusters: int = 6, random_state: int = 42):
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X_df = df[feature_cols].copy()

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_df)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    df_out = df.copy()
    df_out["cluster"] = labels

    return df_out, model, preprocessor, score


def search_best_k(df: pd.DataFrame, k_range=range(3, 11), random_state: int = 42):
    results = []
    for k in k_range:
        clustered_df, _, _, score = fit_kmeans(df, n_clusters=k, random_state=random_state)
        results.append({
            "k": k,
            "silhouette": score,
        })
    return pd.DataFrame(results).sort_values("silhouette", ascending=False)


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = [
        "text_length",
        "text_density",
        "link_count",
        "link_text_ratio",
        "section_count",
        "div_count",
        "li_count",
        "h1_count",
        "h2_count",
        "h3_count",
        "heading_total",
        "js_ui_score",
        "empty_body",
    ]
    return df.groupby("cluster")[agg_cols].mean().round(3)


def outline_patterns(df: pd.DataFrame, top_n: int = 10) -> dict[int, pd.Series]:
    result = {}
    for cluster_id, g in df.groupby("cluster"):
        vc = g["section_outline"].fillna("").value_counts().head(top_n)
        result[int(cluster_id)] = vc
    return result



def main():
    input_path = "output/docomo_faq/structure/page_metrics.csv"
    output_path = "oujtput/docomo_faq/cluster"
    os.makedirs(output_path, exist_ok=True)
    
    df = pd.read_csv("input_path")

    # まず品質確認
    print(df["empty_body"].value_counts(dropna=False))

    # kの候補を見る
    k_result = search_best_k(df, k_range=range(3, 9))
    print(k_result)

    # 例えば上位のk=5を採用
    clustered_df, model, preprocessor, score = fit_kmeans(df, n_clusters=5)
    print("silhouette =", score)

    # 平均特徴を見る
    print(summarize_clusters(clustered_df))

    # section_outline頻出
    patterns = outline_patterns(clustered_df, top_n=5)
    for cluster_id, vc in patterns.items():
        print(f"\n[cluster {cluster_id}]")
        print(vc)

    clustered_df.to_csv("page_clusters.csv", index=False)
    