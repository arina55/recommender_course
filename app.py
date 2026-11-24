import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD, Dataset, Reader

DATA_DIR = Path(__file__).resolve().parent / "data"
VIDEO_GAMES_FILES = (
    "Video_Games_5.jsonl.gz",
    "Video_Games_5.jsonl",
    "Video_Games_5.json.gz",
    "Video_Games_5.json",
)
LEGACY_FILES = ("reviews_Toys_and_Games_5.json.gz", "reviews_Toys_and_Games_5.json")
MODERN_COLUMNS = {
    "user_id": "reviewerID",
    "parent_asin": "asin",
    "rating": "overall",
    "text": "reviewText",
    "title": "summary",
}


def data_path():
    configured = os.environ.get("RECOMMENDER_DATA")
    if configured:
        return Path(configured)
    candidates = [DATA_DIR / name for name in (*VIDEO_GAMES_FILES, *LEGACY_FILES)]
    candidates.extend(DATA_DIR.parent / name for name in LEGACY_FILES)
    return next((path for path in candidates if path.is_file()), candidates[0])


@st.cache_data(show_spinner=False)
def load_data(path, modified_ns):
    # Время изменения входит в ключ кэша при замене файла.
    df = pd.read_json(path, lines=True, dtype=False)
    if set(MODERN_COLUMNS) <= set(df):
        df = df.rename(columns=MODERN_COLUMNS)
    required = ["reviewerID", "asin", "overall", "reviewText", "summary"]
    if not set(required) <= set(df):
        raise ValueError("Ожидаются поля отзывов Amazon Reviews 2023 или Amazon 5-core.")
    df = df[required].copy()
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df = df.dropna(subset=["reviewerID", "asin"])
    df = df[df["overall"].between(1, 5)]
    df = df.drop_duplicates(subset=["reviewerID", "asin"])
    if df.empty:
        raise ValueError("В файле нет подходящих отзывов с оценками от 1 до 5.")
    df[["reviewText", "summary"]] = df[["reviewText", "summary"]].fillna("").astype(str)
    return df


@st.cache_resource(show_spinner=False)
def train_models(path, modified_ns):
    df = load_data(path, modified_ns)
    ratings = Dataset.load_from_df(
        df[["reviewerID", "asin", "overall"]], Reader(rating_scale=(1, 5))
    )
    algo = SVD(n_factors=50, random_state=42)
    algo.fit(ratings.build_full_trainset())

    reviews = df["summary"].str.cat(df["reviewText"], sep=" ")
    item_texts = reviews.groupby(df["asin"]).agg(" ".join)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    if not any(map(tfidf.build_analyzer(), item_texts)):
        raise ValueError("Для TF-IDF в отзывах нужны тексты с распознаваемыми словами.")
    return algo, item_texts.index.tolist(), tfidf.fit_transform(item_texts)


def recommendations(user_id, history, candidates, model, items, matrix):
    n = 50
    svd_rank = sorted(
        candidates, key=lambda item: model.predict(user_id, item).est, reverse=True
    )[:n]
    history_rows = [index for index, item in enumerate(items) if item in history]
    # TF-IDF уже нормализован. Сумма скалярных произведений даёт сумму
    # косинусных сходств без хранения квадратной матрицы товаров.
    profile = np.asarray(matrix[history_rows].sum(axis=0)).ravel()
    similarities = matrix @ profile
    text_rank = [items[index] for index in np.argsort(similarities)[::-1] if items[index] not in history][:n]

    scores = {}
    for ranking in (svd_rank, text_rank):
        for index, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + 0.5 * (n - index)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:10]


def main():
    st.title("🛍️ Рекомендательная система товаров")
    st.write("Ансамбль моделей SVD и TF-IDF")
    path = data_path()
    try:
        with st.spinner("Загружаем отзывы и обучаем модели…"):
            modified_ns = path.stat().st_mtime_ns
            df = load_data(path, modified_ns)
            model, items, matrix = train_models(path, modified_ns)
    except FileNotFoundError:
        st.error(f"Файл отзывов не найден: {path}. Поместите выборку в папку data или задайте RECOMMENDER_DATA.")
        return
    except (OSError, ValueError) as error:
        st.error(f"Не удалось загрузить отзывы или обучить модели: {error}")
        return

    all_users = df["reviewerID"].unique()
    st.success("Модели готовы!")
    user_id = st.selectbox("Выберите ID пользователя:", all_users[:100])
    if st.button("Показать рекомендации"):
        history = set(df.loc[df["reviewerID"] == user_id, "asin"])
        candidates = [item for item in df["asin"].unique() if item not in history]
        ranked_items = recommendations(user_id, history, candidates, model, items, matrix)
        if not ranked_items:
            st.info("Для этого пользователя нет новых товаров.")
            return
        st.subheader("Рекомендации")
        for index, item in enumerate(ranked_items, 1):
            item_info = df[df["asin"] == item]
            rating = item_info["overall"].mean()
            st.write(f"**{index}. Товар {item}** (Средний рейтинг: {rating:.1f} ⭐)")
            title = item_info["summary"].iloc[0]
            if title:
                st.caption(f"Отзыв: {title[:100]}")


if __name__ == "__main__":
    main()
