import pandas as pd
import numpy as np
import pickle
from lightfm import LightFM
from scipy.sparse import load_npz, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Load mô hình và dữ liệu đã huấn luyện ====
model = pickle.load(open("MODEL/lightfm_model.pkl", "rb"))
dataset = pickle.load(open("MODEL/lightfm_dataset.pkl", "rb"))
user_features = load_npz("MODEL/user_features_matrix.npz")
item_features = load_npz("MODEL/item_features_matrix.npz")

test_df = pd.read_csv("Chia_Data/data_test_cold.csv")

# ==== Tiền xử lý dữ liệu test ====
def age_group(age):
    # Nhóm tuổi thành các mốc
    if age <= 25: return "18-25"
    elif age <= 35: return "26-35"
    elif age <= 45: return "36-45"
    elif age <= 60: return "46-60"
    else: return "60+"

test_df["Age_Group"] = test_df["Age"].apply(age_group)
test_df.rename(columns={"Customer_ID": "user_id_raw", "Item_Purchased": "item_id_raw"}, inplace=True)

# ==== Mapping từ dataset để ánh xạ index nội bộ ====
_, user_feat_map, item_id_map, _ = dataset.mapping()
inner_to_item_raw = {v: k for k, v in item_id_map.items()}  # mapping ngược từ index về item raw ID

# ==== Tạo vector đặc trưng người dùng tạm thời cho cold user ====
def build_temp_user_vector(age, gender):
    age_grp = age_group(age)
    features = [f"Gender={gender}", f"Age_Group={age_grp}"]
    feature_index_map = user_feat_map
    indices = [feature_index_map[f] for f in features if f in feature_index_map]
    values = [1.0] * len(indices)
    return csr_matrix((values, ([0]*len(indices), indices)), shape=(1, user_features.shape[1]))

# ==== Đánh giá precision trên từng cold user ====
results = []

item_id_to_index = {v: k for k, v in item_id_map.items()}  # mapping ngược cho item

for user_raw_id in test_df["user_id_raw"].unique():
    user_df = test_df[test_df["user_id_raw"] == user_raw_id]

    # Tạo vector đặc trưng người dùng cold
    age = user_df["Age"].iloc[0]
    gender = user_df["Gender"].iloc[0]
    user_vec = build_temp_user_vector(age, gender)

    # Danh sách item thực tế người đó đã mua
    true_items_raw = user_df["item_id_raw"].unique()
    true_items_inner = [item_id_map[i] for i in true_items_raw if i in item_id_map]
    if not true_items_inner:
        continue

    # Dự đoán điểm cho toàn bộ item
    scores = model.predict(0, np.arange(len(item_id_map)), user_features=user_vec, item_features=item_features)

    # Chọn top-N item có điểm cao nhất, với N = số item thực tế
    top_items_inner = np.argsort(-scores)[:len(true_items_inner)]
    top_items_raw = [inner_to_item_raw[i] for i in top_items_inner]

    # So sánh và tính precision
    correct_items = set(top_items_raw) & set(true_items_raw)
    num_correct = len(correct_items)
    precision = num_correct / len(true_items_inner)

    results.append({
        "Customer_ID": user_raw_id,
        "Số SP Mua Thực Tế": len(true_items_raw),
        "Số SP Được Gợi Ý": len(top_items_raw),
        "Số SP Đúng": num_correct,
        "Precision": precision,
        "SP Đã Mua": ", ".join(true_items_raw),
        "SP Được Gợi Ý": ", ".join(top_items_raw),
        "SP Đúng": ", ".join(correct_items)
    })

# ==== Xuất kết quả thành CSV ====
results_df = pd.DataFrame(results)
results_df.to_csv("Ket_Qua_Test_Cold.csv", index=False, encoding="utf-8-sig")

print("✅ Đã lưu file 'Ket_Qua_Test_Cold.csv' với chi tiết sản phẩm đúng.")
print("🎯 Precision trung bình toàn bộ người dùng:", results_df["Precision"].mean())

# ==== Vẽ biểu đồ phân bố Precision ====
results_df_sorted = results_df.sort_values(by="Precision", ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 6))
sns.barplot(x=results_df_sorted.index, y=results_df_sorted["Precision"], color="skyblue")
plt.axhline(results_df_sorted["Precision"].mean(), color="red", linestyle="--", label="Precision trung bình")
plt.title("Phân bố Precision theo người dùng (Test Cold Start)")
plt.xlabel("Người dùng (đã sắp xếp theo Precision)")
plt.ylabel("Precision")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("Bieu_Do_Test_Cold.png")
print("✅ Đã lưu biểu đồ tại Bieu_Do_Test_Cold.png")