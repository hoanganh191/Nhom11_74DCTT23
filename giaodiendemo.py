import pandas as pd
import numpy as np
import pickle
from lightfm import LightFM
from scipy.sparse import load_npz
import gradio as gr
import os
import base64

# === Load mô hình và dữ liệu đã train ===
model = pickle.load(open("MODEL/lightfm_model.pkl", "rb"))
dataset = pickle.load(open("MODEL/lightfm_dataset.pkl", "rb"))
user_features = load_npz("MODEL/user_features_matrix.npz")
item_features = load_npz("MODEL/item_features_matrix.npz")

# === Load lại file data_train.csv ===
df_train = pd.read_csv("Chia_Data/data_train.csv")
df_train["Customer_ID"] = df_train["Customer_ID"].astype(str)
df_train = df_train.drop_duplicates(subset="Customer_ID")
allowed_ids = set(df_train["Customer_ID"].unique())

# === Mapping từ mô hình ===
user_id_map, _, item_id_map, _ = dataset.mapping()
item_id_reverse = {v: k for k, v in item_id_map.items()}
user_id_map = {str(k): v for k, v in user_id_map.items()}

# === Hàm render HTML box có ảnh base64 và căn giữa ảnh ===
def render_item_boxes(items):
    box_html = ""
    for item in items:
        image_path = f"img/{item}.png"
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            img_tag = f'<img src="data:image/png;base64,{encoded}" alt="{item}" style="width:100px;height:100px;object-fit:contain;display:block;margin:0 auto;">'
        else:
            img_tag = f'<img src="https://via.placeholder.com/100?text=No+Image" alt="{item}" style="width:100px;height:100px;object-fit:contain;display:block;margin:0 auto;">'

        box_html += f"""
        <div style="
            background: #f0f4ff;
            border-radius: 12px;
            padding: 15px;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            width: 200px;
            display: inline-block;
        ">
            {img_tag}<br>
            <h4 style="color: #2a6df4; margin-top: 10px;">{item}</h4>
        </div>
        """
    return box_html

# === Hàm gợi ý sản phẩm ===
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

    result_html = render_item_boxes(recommended_items)
    return user_info_text, result_html

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

    result_output = gr.HTML()

    btn.click(fn=recommend_products,
              inputs=[customer_input, topn_slider],
              outputs=[user_info_output, result_output])

# === Chạy Gradio ===
demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
