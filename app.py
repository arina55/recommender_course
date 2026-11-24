import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import gzip
import shutil
import os

# --- 1. ЗАГРУЗКА ДАННЫХ (Кэшируем, чтобы не скачивать каждый раз) ---
@st.cache_data
def load_data():
    url = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz"
    gz_filename = "reviews_Toys_and_Games_5.json.gz"
    json_filename = "reviews_Toys_and_Games_5.json"
    
    if not os.path.exists(json_filename):
        urllib.request.urlretrieve(url, gz_filename)
        with gzip.open(gz_filename, 'rb') as f_in:
            with open(json_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    df = pd.read_json(json_filename, lines=True)
    df_clean = df.drop_duplicates(subset=['reviewerID', 'asin'])
    df_clean = df_clean[['reviewerID', 'asin', 'overall', 'reviewText', 'summary']]
    return df_clean

@st.cache_resource
def train_models(df_clean):
    # Обучение SVD
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_clean[['reviewerID', 'asin', 'overall']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=50, random_state=42)
    algo.fit(trainset)
    
    # Обучение TF-IDF
    item_texts = df_clean.groupby('asin')['reviewText'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(item_texts['reviewText'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_texts['asin'])}
    
    return algo, item_texts, cosine_sim, item_id_to_index

# --- 2. ФУНКЦИИ РЕКОМЕНДАЦИЙ ---
def get_svd_recs(user_id, algo, all_items, n=50):
    predictions = []
    for item in all_items:
        pred = algo.predict(user_id, item)
        predictions.append((item, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in predictions[:n]]

def get_content_recs(user_history, item_id_to_index, cosine_sim, all_items, n=50):
    user_indices = [item_id_to_index[item] for item in user_history if item in item_id_to_index]
    if not user_indices:
        return []
    sim_scores = np.zeros(len(item_id_to_index))
    for idx in user_indices:
        sim_scores += cosine_sim[idx]
    sim_scores_indices = np.argsort(sim_scores)[::-1]
    
    recommended = []
    for i in sim_scores_indices:
        item_id = list(item_id_to_index.keys())[i]
        if item_id not in user_history:
            recommended.append(item_id)
        if len(recommended) == n:
            break
    return recommended

# --- 3. ИНТЕРФЕЙС ---
st.title("🛍️ Рекомендательная система товаров")
st.write("Курсовая работа: Ансамбль моделей (SVD + TF-IDF)")

with st.spinner("Загружаем данные и обучаем модели (займет ~1 минуту)..."):
    df_clean = load_data()
    algo, item_texts, cosine_sim, item_id_to_index = train_models(df_clean)
    all_unique_items = df_clean['asin'].unique().tolist()
    all_users = df_clean['reviewerID'].unique().tolist()

st.success("Модели готовы!")

# Выбор пользователя
user_id = st.selectbox("Выберите ID пользователя:", all_users[:100])

if st.button("Показать рекомендации"):
    with st.spinner("Считаем..."):
        user_history = df_clean[df_clean['reviewerID'] == user_id]['asin'].tolist()
        
        svd_recs = get_svd_recs(user_id, algo, all_unique_items, n=50)
        tfidf_recs = get_content_recs(user_history, item_id_to_index, cosine_sim, all_unique_items, n=50)
        
        alpha = 0.5
        scores = {}
        for i, item in enumerate(svd_recs):
            scores[item] = scores.get(item, 0) + alpha * (50 - i)
        for i, item in enumerate(tfidf_recs):
            scores[item] = scores.get(item, 0) + (1 - alpha) * (50 - i)
            
        final_recs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:10]
        
        st.write(f"### Топ-10 рекомендаций для пользователя {user_id}:")
        
        # Выводим в виде красивого списка
        for i, item in enumerate(final_recs, 1):
            # Получаем название/описание товара, если есть
            item_info = df_clean[df_clean['asin'] == item]
            if not item_info.empty:
                title = item_info['summary'].iloc[0]
                rating = item_info['overall'].mean()
                st.write(f"**{i}. Товар {item}** (Средний рейтинг: {rating:.1f}⭐)")
                if title and title != "nan":
                    st.write(f"   *{title[:100]}...*")
            else:
                st.write(f"**{i}. Товар {item}**")