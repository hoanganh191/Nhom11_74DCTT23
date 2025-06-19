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

# === Load lại đúng file data_train.csv hiện tại để kiểm tra ID gốc và thông tin user ===
df_train = pd.read_csv("Chia_Data/data_train.csv")
df_train["Customer_ID"] = df_train["Customer_ID"].astype(str)
df_train = df_train.drop_duplicates(subset="Customer_ID")  # mỗi ID chỉ 1 dòng thông tin
allowed_ids = set(df_train["Customer_ID"].unique())

# === Mapping từ mô hình ===
user_id_map, _, item_id_map, _ = dataset.mapping()
item_id_reverse = {v: k for k, v in item_id_map.items()}
user_id_map = {str(k): v for k, v in user_id_map.items()}

# === Hàm gợi ý sản phẩm, trả thông tin user + danh sách sản phẩm ===
def recommend_products(customer_id, top_n=5):
    customer_id = str(customer_id).strip()

    if customer_id not in allowed_ids:
        return f"❌ Customer_ID '{customer_id}' không có trong dữ liệu huấn luyện.", ""

    if customer_id not in user_id_map:
        return f"❌ Customer_ID '{customer_id}' không tồn tại trong mô hình.", ""

    user_info = df_train[df_train["Customer_ID"] == customer_id][["Gender", "Age"]].iloc[0]
    gender = user_info["Gender"]
    age = user_info["Age"]
    user_info_text = f"👤 Giới tính: {gender}\n🎂 Tuổi: {age}"

    user_index = user_id_map[customer_id]
    scores = model.predict(user_index, np.arange(len(item_id_map)),
                           user_features=user_features, item_features=item_features)
    top_items = np.argsort(-scores)[:top_n]
    recommended_items = [item_id_reverse[i] for i in top_items]

    result_text = "\n".join([f"👉 {item}" for item in recommended_items])
    return user_info_text, result_text

# === Giao diện Gradio ===
with gr.Blocks(title="Gợi ý sản phẩm cho khách hàng") as demo:
    gr.Markdown("# 🎯 Gợi ý sản phẩm dựa trên Customer_ID")
    with gr.Row():
        with gr.Column():
            customer_input = gr.Textbox(label="Nhập Customer_ID")
            topn_slider = gr.Slider(1, 20, value=5, step=1, label="Số sản phẩm gợi ý")
            btn = gr.Button("🔍 Gợi ý ngay")
        with gr.Column():
            user_info_output = gr.Textbox(label="Thông tin người dùng", lines=2)

    result_output = gr.Textbox(label="Sản phẩm gợi ý", lines=10)

    btn.click(fn=recommend_products,
              inputs=[customer_input, topn_slider],
              outputs=[user_info_output, result_output])

    demo.launch()