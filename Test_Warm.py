import pandas as pd
import numpy as np
import pickle
from lightfm import LightFM
from scipy.sparse import load_npz

# ==== Load mô hình và dữ liệu ====
model = pickle.load(open("MODEL/lightfm_model.pkl", "rb"))
dataset = pickle.load(open("MODEL/lightfm_dataset.pkl", "rb"))
user_features = load_npz("MODEL/user_features_matrix.npz")
item_features = load_npz("MODEL/item_features_matrix.npz")

test_df = pd.read_csv("Chia_Data/data_test_warm.csv")

# ==== Tiền xử lý ====
def age_group(age):
    if age <= 25: return "18-25"
    elif age <= 35: return "26-35"
    elif age <= 45: return "36-45"
    elif age <= 60: return "46-60"
    else: return "60+"

test_df["Age_Group"] = test_df["Age"].apply(age_group)
test_df.rename(columns={"Customer_ID": "user_id_raw", "Item_Purchased": "item_id_raw"}, inplace=True)

# ==== Mapping từ dataset ====
user_id_map, _, item_id_map, _ = dataset.mapping()
inner_to_item_raw = {v: k for k, v in item_id_map.items()}

# ==== Tính precision và lưu kết quả ====
results = []

for user_raw_id in test_df["user_id_raw"].unique():
    if user_raw_id not in user_id_map:
        continue

    uid = user_id_map[user_raw_id]

    # Các sản phẩm đã mua
    true_items_raw = test_df[test_df["user_id_raw"] == user_raw_id]["item_id_raw"].unique()
    true_items_inner = [item_id_map[i] for i in true_items_raw if i in item_id_map]
    if not true_items_inner:
        continue

    # Dự đoán tất cả item
    scores = model.predict(uid, np.arange(len(item_id_map)), user_features=user_features, item_features=item_features)

    # Top-N sản phẩm gợi ý theo số sản phẩm thực tế
    top_items_inner = np.argsort(-scores)[:len(true_items_inner)]
    top_items_raw = [inner_to_item_raw[i] for i in top_items_inner]

    # So sánh
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

# ==== Xuất ra CSV đẹp ====
results_df = pd.DataFrame(results)
results_df.to_csv("Ket_Qua_Test_Warm.csv", index=False, encoding="utf-8-sig")

print("✅ Đã lưu file 'Ket_Qua_Test_Warm' với chi tiết sản phẩm đúng.")
print("🎯 Precision trung bình toàn bộ người dùng:", results_df["Precision"].mean())
