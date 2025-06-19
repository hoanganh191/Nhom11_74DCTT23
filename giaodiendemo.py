import pandas as pd
import numpy as np
import pickle
from lightfm import LightFM
from scipy.sparse import load_npz
import gradio as gr

# === Load mô hình và dữ liệu đã train ===
model = pickle.load(open("MODEL/lightfm_model.pkl", "rb"))
dataset = pickle.load(open("MODEL/lightfm_dataset.pkl", "rb"))
user_features = load_npz("MODEL/user_features_matrix.npz")
item_features = load_npz("MODEL/item_features_matrix.npz")

# === Load dữ liệu test để lấy danh sách Customer_ID gốc ===
# === Load dữ liệu TRAIN (đã dùng để huấn luyện) ===
df = pd.read_csv("Chia_Data/data_train.csv")
df["Customer_ID"] = df["Customer_ID"].astype(str)
df["Item_Purchased"] = df["Item_Purchased"].astype(str)
df.rename(columns={"Customer_ID": "user_id_raw", "Item_Purchased": "item_id_raw"}, inplace=True)


# === Mapping từ dataset ===
user_id_map, _, item_id_map, _ = dataset.mapping()
item_id_reverse = {v: k for k, v in item_id_map.items()}

# Ép user_id_map về chuỗi để đồng bộ
user_id_map = {str(k): v for k, v in user_id_map.items()}

# === Hàm gợi ý sản phẩm ===
def recommend_products(customer_id, top_n=5):
    customer_id = str(customer_id).strip()

    if customer_id not in user_id_map:
        return f"❌ Customer_ID '{customer_id}' không tồn tại trong hệ thống."

    user_index = user_id_map[customer_id]
    scores = model.predict(user_ids=user_index,
                           item_ids=np.arange(len(item_id_map)),
                           user_features=user_features,
                           item_features=item_features)

    top_items = np.argsort(-scores)[:top_n]
    recommended_items = [item_id_reverse[i] for i in top_items]

    return recommended_items

# === Giao diện Gradio ===
gr.Interface(
    fn=recommend_products,
    inputs=[
        gr.Textbox(label="Nhập Customer_ID"),
        gr.Slider(1, 20, value=5, step=1, label="Số sản phẩm gợi ý")
    ],
    outputs="text",
    title="🔍 Gợi ý sản phẩm với LightFM",
    description="Nhập Customer_ID để nhận sản phẩm phù hợp từ mô hình đã huấn luyện."
).launch()
