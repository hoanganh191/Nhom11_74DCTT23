# Cần có 2 thư viện này trước khi train và được train ở Google Colab
# !pip install lightfm
# !pip install pandas scikit-learn

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from scipy.sparse import save_npz
import pickle
import os

# ===== 1. Đọc dữ liệu =====
df_train = pd.read_csv('/content/data_train.csv')
df_val = pd.read_csv('/content/data_val.csv')

# ===== 2. Gộp dữ liệu để fit encoder =====
df_all = pd.concat([df_train, df_val])

# ===== 3. Tạo cột Age_Group =====
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

for df in [df_all, df_train, df_val]:
    df["Age_Group"] = df["Age"].apply(age_group)

# ===== 4. Chuẩn hoá tên cột =====
for df in [df_all, df_train, df_val]:
    df.rename(columns={
        "Customer_ID": "user_id_raw",
        "Item_Purchased": "item_id_raw"
    }, inplace=True)

# ===== 5. Khởi tạo Dataset và fit =====
dataset = Dataset()
#Tập Hợp đặc trưng người dùng
user_features_list = list(f"Gender={g}" for g in df_all["Gender"].unique()) + \
                     list(f"Age_Group={a}" for a in df_all["Age_Group"].unique())

#Tập hợp đặc trưng sản phẩm
item_features_list = list(f"Category={c}" for c in df_all["Category"].unique()) + \
                     list(f"Season={s}" for s in df_all["Season"].unique())

#Chuẩn hóa để chuẩn bị tạo thành ma trận
dataset.fit(
    users=df_all["user_id_raw"],
    items=df_all["item_id_raw"],
    user_features=user_features_list,
    item_features=item_features_list
)

# ===== 6. Xây dựng user/item features =====
#Tổng hợp tất cả các giá trị có trong data của mỗi đặc trưng

def build_user_features(df):
    rows = df.drop_duplicates("user_id_raw")
    features = [
        (row["user_id_raw"], [f"Gender={row['Gender']}", f"Age_Group={row['Age_Group']}"])
        for _, row in rows.iterrows()
    ]
    return dataset.build_user_features(features)

def build_item_features(df):
    rows = df.drop_duplicates("item_id_raw")
    features = [
        (row["item_id_raw"], [f"Category={row['Category']}", f"Season={row['Season']}"])
        for _, row in rows.iterrows()
    ]
    return dataset.build_item_features(features)

#Xây ma trận từ đặc trưng
user_features_matrix = build_user_features(df_all)
item_features_matrix = build_item_features(df_all)

# ===== 7. Tạo ma trận interactions ===== ma trận tương tác
def build_interactions(df):
    return dataset.build_interactions([
        (row["user_id_raw"], row["item_id_raw"], row["Review_Rating"])
        for _, row in df.iterrows()
    ])[0]

interactions_train = build_interactions(df_train)
interactions_val = build_interactions(df_val)

# ===== 8. Huấn luyện mô hình =====

#Tạo mô hình LightFM
model = LightFM(
    loss="warp", #Hàm mất mát để học xếp hạng ,Ưu tiên tính điểm cao hơn cho người dùng đã tương tác WARP = Weighted Approximate-Rank Pairwise
    no_components=50, #Số chiều của vector ẩn
    learning_rate=0.01, #Tốc độ học
    item_alpha=1e-6, #Chúng là hệ số regularization (chuẩn hoá) để Dùng để chống overfitting trong quá trình huấn luyện mô hình ,Ngăn vector sản phẩm học quá “mạnh” hoặc “lớn”.
    user_alpha=1e-6 #Chúng là hệ số regularization (chuẩn hoá) để Dùng để chống overfitting trong quá trình huấn luyện mô hình, Ngăn vector người dùng học quá “kỹ” hoặc “phức tạp”.
    # WARP cho mô hình lý do để học tốt,
    # còn item_alpha và user_alpha giúp mô hình học an toàn.
)

#Huấn Luyện Mô hình
model.fit(
    interactions=interactions_train,
    user_features=user_features_matrix,
    item_features=item_features_matrix,
    epochs=60, #Số vòng lặp học
    num_threads=4 #Số luồng xử lý đồng thời
)

# ===== 9. Đánh giá trên tập validation =====
print("✅ Validation:")
print("Precision@5:", precision_at_k(model, interactions_val, user_features=user_features_matrix, item_features=item_features_matrix, k=5).mean())
print("AUC:", auc_score(model, interactions_val, user_features=user_features_matrix, item_features=item_features_matrix).mean())

# ===== 10. Lưu mô hình và dữ liệu =====
os.makedirs("MODEL", exist_ok=True)

with open("MODEL/lightfm_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("MODEL/lightfm_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

save_npz("MODEL/train_interactions.npz", interactions_train)
save_npz("MODEL/val_interactions.npz", interactions_val)
save_npz("MODEL/user_features_matrix.npz", user_features_matrix)
save_npz("MODEL/item_features_matrix.npz", item_features_matrix)

print("✅ Đã lưu mô hình và dữ liệu vào thư mục MODEL/")