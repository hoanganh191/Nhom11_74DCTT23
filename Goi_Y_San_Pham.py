import pickle
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
import pandas as pd
#demo github 
# Đọc dữ liệu từ file CSV chứa thông tin sản phẩm
items = pd.read_csv("Data_Split/test_lightfm.csv")  # Giả sử đây là file chứa thông tin sản phẩm

# Load mô hình
with open("MODEL/lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
with open("MODEL/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Load user/item features
user_features = sp.load_npz("MODEL/user_features_matrix.npz")
item_features = sp.load_npz("MODEL/item_features_matrix.npz")

def recommend_for_user_id(user_id, top_n=5):
    item_index_map = {v: k for k, v in dataset.mapping()[2].items()}
    user_index_map = dataset.mapping()[0]

    if user_id not in user_index_map:
        print(f"User ID '{user_id}' không tồn tại trong dữ liệu.")
        return []

    internal_user_id = user_index_map[user_id]

    scores = model.predict(
        user_ids=internal_user_id,
        item_ids=np.arange(item_features.shape[0]),
        user_features=user_features,
        item_features=item_features
    )

    top_items_indices = np.argsort(-scores)[:top_n]
    recommendations = []

    for idx in top_items_indices:
        if idx in item_index_map:
            item_name = item_index_map[idx]
            if item_name in items['Item Purchased'].values:
                item_info = items[items['Item Purchased'] == item_name].iloc[0]
                recommendations.append({
                    "Product": item_name,
                    "Category": item_info['Category'],
                    "Size": item_info['Size'],
                    "Color": item_info['Color']
                })
    return recommendations

def recommend_for_new_user(age, gender, location, top_n=5):
    new_user_features_list = [str(age), gender, location]
    user_feature_map = dataset.mapping()[1]
    num_total_user_features = user_features.shape[1]

    new_user_features_lil = sp.lil_matrix((1, num_total_user_features))

    for feature in new_user_features_list:
        if feature in user_feature_map:
            feature_idx = user_feature_map[feature]
            new_user_features_lil[0, feature_idx] = 1
        else:
            print(f"⚠️ Warning: '{feature}' không nằm trong dữ liệu huấn luyện. Bỏ qua.")

    new_user_features_matrix = new_user_features_lil.tocsr()
    item_index_map = {v: k for k, v in dataset.mapping()[2].items()}

    scores = model.predict(
        user_ids=0,
        item_ids=np.arange(item_features.shape[0]),
        user_features=new_user_features_matrix,
        item_features=item_features
    )

    top_items_indices = np.argsort(-scores)[:top_n]
    recommendations = []

    for idx in top_items_indices:
        if idx in item_index_map:
            item_name = item_index_map[idx]
            if item_name in items['Item Purchased'].values:
                item_info = items[items['Item Purchased'] == item_name].iloc[0]
                recommendations.append({
                    "Product": item_name,
                    "Category": item_info['Category'],
                    "Size": item_info['Size'],
                    "Color": item_info['Color']
                })
    return recommendations

# ========== Giao diện người dùng ==========
def main():
    print("===== Hệ thống Gợi ý Sản phẩm =====")
    print("1. Nhập thông tin người dùng mới")
    print("2. Tìm theo ID người dùng có sẵn")
    choice = input("Chọn (1 hoặc 2): ")

    match choice:
        case "1":
            try:
                age = int(input("Nhập tuổi: "))
                gender = input("Nhập giới tính (Male/Female): ")
                location = input("Nhập địa điểm: ")
                results = recommend_for_new_user(age, gender, location)
                if results:
                    print("🔎 Gợi ý sản phẩm:")
                    for r in results:
                        print(r)
                else:
                    print("⚠️ Không có gợi ý nào.")
            except Exception as e:
                print(f"Lỗi: {e}")
        case "2":
            user_id = int(input("Nhập user ID : "))
            results = recommend_for_user_id(user_id)
            if results:
                print("🔎 Gợi ý sản phẩm:")
                for r in results:
                    print(r)
            else:
                print("⚠️ Không có gợi ý nào.")
        case _:
            print("⚠️ Lựa chọn không hợp lệ. Vui lòng chọn 1 hoặc 2.")

if __name__ == "__main__":
    main()
