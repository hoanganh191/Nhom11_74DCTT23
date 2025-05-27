import pandas as pd
import numpy as np
import pickle
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import load_npz

# === Load mô hình và dữ liệu đã lưu ===
with open("MODEL/lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("MODEL/lightfm_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

user_features = load_npz("MODEL/user_features_matrix.npz")
item_features = load_npz("MODEL/item_features_matrix.npz")

# === Load dữ liệu test gốc ===
test_df = pd.read_csv("Chia_Data/data_test.csv")

def age_group(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 45:
        return "36-45"
    elif age <= 60:
        return "46-60"
    else:
        return "60+"

test_df["Age_Group"] = test_df["Age"].apply(age_group)
test_df.rename(columns={"Customer_ID": "user_id_raw", "Item_Purchased": "item_id_raw"}, inplace=True)

# Mapping
user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
inv_user_map = {v: k for k, v in user_id_map.items()}
inv_item_map = {v: k for k, v in item_id_map.items()}

unique_user_ids = test_df["user_id_raw"].unique()
n_items = len(item_id_map)

# Dự đoán
results = []
for user_raw in unique_user_ids:
    if user_raw not in user_id_map:
        continue
    uid = user_id_map[user_raw]
    scores = model.predict(uid, np.arange(n_items), user_features=user_features, item_features=item_features)
    top_items = np.argsort(-scores)[:5]
    recommended = [inv_item_map[i] for i in top_items]

    true_items = test_df[test_df["user_id_raw"] == user_raw]["item_id_raw"].tolist()

    hits = len(set(recommended) & set(true_items))
    precision = hits / len(recommended)
    recall = hits / len(true_items) if len(true_items) > 0 else 0
    hit = 1 if hits > 0 else 0

    results.append({
        "Customer_ID": user_raw,
        "True_Items": ",".join(true_items),
        "Recommended_Items": ",".join(recommended),
        "Hit@5": hit,
        "Precision@5": round(precision, 4),
        "Recall@5": round(recall, 4)
    })

# Xuất kết quả
result_df = pd.DataFrame(results)
result_df.to_csv("prediction_vs_actual.csv", index=False)
print("✅ Đã lưu kết quả vào prediction_vs_actual.csv")
