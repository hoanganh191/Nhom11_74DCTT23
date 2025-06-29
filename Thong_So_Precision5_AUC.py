import pickle
import numpy as np
import pandas as pd
from lightfm.evaluation import precision_at_k, auc_score  # ✅ Thêm auc_score
from scipy.sparse import load_npz

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
user_features_matrix = load_npz("MODEL/user_features_matrix.npz")
item_features_matrix = load_npz("MODEL/item_features_matrix.npz")

# ===== 3. Hàm đánh giá Precision@5 và AUC =====
def evaluate_model(test_file_path, label):
    test_df = pd.read_csv(test_file_path)

    # Tạo cột Age_Group
    test_df["Age_Group"] = test_df["Age"].apply(age_group)

    # Đổi tên cột
    test_df.rename(columns={
        "Customer_ID": "user_id_raw",
        "Item_Purchased": "item_id_raw"
    }, inplace=True)

    # Tạo ma trận interactions
    test_interactions, _ = dataset.build_interactions([
        (row["user_id_raw"], row["item_id_raw"], row["Review_Rating"])
        for _, row in test_df.iterrows()
    ])

    # Tính Precision@5
    precision = precision_at_k(
        model,
        test_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        k=5
    ).mean()

    # Tính AUC
    auc = auc_score(
        model,
        test_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix
    ).mean()

    print(f"📁 Đánh giá trên {label}")
    print(f"🎯 Precision@5: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"📈 AUC: {auc:.4f} ({auc * 100:.2f}%)\n")

# ===== 4. Chạy đánh giá =====
evaluate_model("Chia_Data/data_test_cold.csv", "data_test_cold.csv")
evaluate_model("Chia_Data/data_test_warm.csv", "data_test_warm.csv")
