# Cần có 2 thư viện này trước khi train
# !pip install lightfm
# !pip install pandas scikit-learn

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score

# Đọc dữ liệu
train_df = pd.read_csv('/content/data_train.csv')
val_df = pd.read_csv('/content/data_val.csv')
test_df = pd.read_csv('/content/data_test.csv')

# Gộp dữ liệu để chuẩn hóa và mã hóa
full_df = pd.concat([train_df, val_df, test_df])

# ======= 3. Tạo Age_Group =======
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

full_df["Age_Group"] = full_df["Age"].apply(age_group)
train_df["Age_Group"] = train_df["Age"].apply(age_group)
val_df["Age_Group"] = val_df["Age"].apply(age_group)
test_df["Age_Group"] = test_df["Age"].apply(age_group)

# ======= 4. Đổi tên cột cho nhất quán =======
for df in [full_df, train_df, val_df, test_df]:
    df.rename(columns={
        "Customer_ID": "user_id_raw",
        "Item_Purchased": "item_id_raw"
    }, inplace=True)

# ======= 5. Khởi tạo Dataset và fit =======
dataset = Dataset()
user_features = list(f"Gender={g}" for g in full_df["Gender"].unique()) + \
                list(f"Age_Group={a}" for a in full_df["Age_Group"].unique())

item_features = list(f"Category={c}" for c in full_df["Category"].unique()) + \
                list(f"Season={s}" for s in full_df["Season"].unique())

dataset.fit(
    users=full_df["user_id_raw"],
    items=full_df["item_id_raw"],
    user_features=user_features,
    item_features=item_features
)

# ======= 6. Build user/item features =======
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

user_features_matrix = build_user_features(full_df)
item_features_matrix = build_item_features(full_df)

# ======= 7. Build interaction matrices =======
def build_interactions(df):
    return dataset.build_interactions([
        (row["user_id_raw"], row["item_id_raw"], row["Review_Rating"])
        for _, row in df.iterrows()
    ])[0]

train_interactions = build_interactions(train_df)
val_interactions = build_interactions(val_df)
test_interactions = build_interactions(test_df)

# ======= 8. Huấn luyện mô hình với thông số đã điều chỉnh =======
model = LightFM(
    loss="warp",               # hàm mất mát phù hợp với ranking
    no_components=50,          # tăng số chiều embedding
    learning_rate=0.01,        # giảm tốc độ học để ổn định hơn
    item_alpha=1e-6,           # regularization cho item
    user_alpha=1e-6            # regularization cho user
)

model.fit(
    interactions=train_interactions,
    user_features=user_features_matrix,
    item_features=item_features_matrix,
    epochs=60,                 # tăng số epoch để học kỹ hơn
    num_threads=4
)

 #======= 9. Đánh giá mô hình =======
print("✅ Validation:")
print("Precision@5:", precision_at_k(model, val_interactions, user_features=user_features_matrix, item_features=item_features_matrix, k=5).mean())
print("AUC:", auc_score(model, val_interactions, user_features=user_features_matrix, item_features=item_features_matrix).mean())

print("✅ Test:")
print("Precision@5:", precision_at_k(model, test_interactions, user_features=user_features_matrix, item_features=item_features_matrix, k=5).mean())
print("AUC:", auc_score(model, test_interactions, user_features=user_features_matrix, item_features=item_features_matrix).mean())

#==10 Lưu kết quả huấn luyện ===
import os
import pickle
import numpy as np
from scipy.sparse import save_npz  # ✅ Thêm dòng này

# Tạo thư mục nếu chưa tồn tại
os.makedirs("KETQUA", exist_ok=True)

# Lưu mô hình LightFM
with open("KETQUA/lightfm_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Lưu dataset đã fit
with open("KETQUA/lightfm_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

# Lưu các ma trận interactions (nếu là sparse)
save_npz("KETQUA/train_interactions.npz", train_interactions)
save_npz("KETQUA/val_interactions.npz", val_interactions)
save_npz("KETQUA/test_interactions.npz", test_interactions)

# Lưu user/item features matrix bằng scipy.sparse.save_npz ✅
save_npz("KETQUA/user_features_matrix.npz", user_features_matrix)
save_npz("KETQUA/item_features_matrix.npz", item_features_matrix)

print("✅ Đã lưu toàn bộ mô hình và dữ liệu vào thư mục KETQUA/")
