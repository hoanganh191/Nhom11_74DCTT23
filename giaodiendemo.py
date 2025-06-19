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

# === Load lại đúng file data_train.csv hiện tại để kiểm tra ID gốc ===
df_train = pd.read_csv("Chia_Data/data_train.csv")
df_train["Customer_ID"] = df_train["Customer_ID"].astype(str)
allowed_ids = set(df_train["Customer_ID"].unique())  # Tập hợp các ID thật sự có trong file

# === Mapping từ mô hình ===
user_id_map, _, item_id_map, _ = dataset.mapping()
item_id_reverse = {v: k for k, v in item_id_map.items()}
user_id_map = {str(k): v for k, v in user_id_map.items()}  # Đảm bảo key là string

# === Hàm gợi ý sản phẩm ===
def recommend_products(customer_id, top_n=5):
    customer_id = str(customer_id).strip()

    if customer_id not in allowed_ids:
        return f"❌ Customer_ID '{customer_id}' không có trong file train hiện tại."

    if customer_id not in user_id_map:
        return f"❌ Customer_ID '{customer_id}' không được ánh xạ trong mô hình."

    user_index = user_id_map[customer_id]
    scores = model.predict(user_index, np.arange(len(item_id_map)),
                           user_features=user_features, item_features=item_features)

    top_items = np.argsort(-scores)[:top_n]
    return [item_id_reverse[i] for i in top_items]

# === Giao diện Gradio ===
gr.Interface(
    fn=recommend_products,
    inputs=[
        gr.Textbox(label="Nhập Customer_ID"),
        gr.Slider(1, 20, value=5, step=1, label="Số sản phẩm gợi ý")
    ],
    outputs="text",
    title="🔍 Gợi ý sản phẩm LightFM",
    description="Chỉ gợi ý cho Customer_ID có trong data_train.csv hiện tại."
).launch()
