import pickle
import numpy as np
import pandas as pd
from lightfm.evaluation import precision_at_k

# ===== Hàm hỗ trợ =====
def load_npz_object(path):
    return np.load(path, allow_pickle=True)["data"].item()

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

# ===== 1. Load mô hình và dataset đã lưu =====
with open("MODEL/lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("MODEL/lightfm_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# ===== 2. Load ma trận feature =====
user_features_matrix = load_npz_object("MODEL/user_features_matrix.npz")
item_features_matrix = load_npz_object("MODEL/item_features_matrix.npz")

# ===== 3. Load file test và xử lý =====
test_df = pd.read_csv("Chia_Data/data_test.csv")

# Tạo cột Age_Group
test_df["Age_Group"] = test_df["Age"].apply(age_group)

# Đổi tên cột cho khớp
test_df.rename(columns={
    "Customer_ID": "user_id_raw",
    "Item_Purchased": "item_id_raw"
}, inplace=True)

# ===== 4. Tạo ma trận test_interactions =====
test_interactions, _ = dataset.build_interactions([
    (row["user_id_raw"], row["item_id_raw"], row["Review_Rating"])
    for _, row in test_df.iterrows()
])

# ===== 5. Tính Precision@5 =====
precision = precision_at_k(
    model,
    test_interactions,
    user_features=user_features_matrix,
    item_features=item_features_matrix,
    k=5
).mean()

print(f"🎯 Precision@5 trên file data_test.csv: {precision:.4f} ({precision * 100:.2f}%)")
