import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD

# 1. ЗАГРУЗКА ДАННЫХ
print("1. Скачиваем данные...")
url = "https://raw.githubusercontent.com/Julia-Koval/Amazon-Reviews-Dataset/master/Video_Games_5.json"
filename = "amazon_data.json"
urllib.request.urlretrieve(url, filename)
df = pd.read_json(filename, lines=True)

# 2. ЧИСТКА
print("2. Чистим данные...")
df_clean = df.drop_duplicates(subset=['reviewerID', 'asin'])
df_clean = df_clean[['reviewerID', 'asin', 'overall', 'reviewText', 'summary']]

# 3. РАЗБИЕНИЕ
print("3. Делим на train/test...")
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)
test_users = test_df['reviewerID'].unique()[:100]

# 4. МЕТРИКА
def precision_at_k(recommended_items, actual_items, k=10):
    top_k = recommended_items[:k]
    hits = len(set(top_k) & set(actual_items))
    return hits / k

# 5. BASELINE
print("4. Считаем Baseline...")
item_stats = df_clean.groupby('asin').agg(avg_rating=('overall', 'mean'), review_count=('overall', 'count')).reset_index()
item_stats = item_stats[item_stats['review_count'] >= 10]
top_popular_items = item_stats.sort_values(by=['avg_rating', 'review_count'], ascending=[False, False])
baseline_recommendations = top_popular_items.head(10)['asin'].tolist()

# 6. SVD
print("5. Обучаем SVD...")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[['reviewerID', 'asin', 'overall']], reader)
trainset = data.build_full_trainset()
algo = SVD(n_factors=50, random_state=42)
algo.fit(trainset)

print("6. Считаем точность SVD...")
def get_svd_recommendations(user_id, algo, all_items, n=10):
    predictions = []
    for item in all_items:
        pred = algo.predict(user_id, item)
        predictions.append((item, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in predictions[:n]]

all_unique_items = df_clean['asin'].unique().tolist()

svd_precisions = []
for user in test_users:
    actual_items = test_df[test_df['reviewerID'] == user]['asin'].tolist()
    svd_recs = get_svd_recommendations(user, algo, all_unique_items, n=10)
    svd_precisions.append(precision_at_k(svd_recs, actual_items, k=10))

avg_svd_precision = sum(svd_precisions) / len(svd_precisions)
print(f"=== СРЕДНИЙ PRECISION@10 ДЛЯ SVD: {avg_svd_precision:.4f} ===")