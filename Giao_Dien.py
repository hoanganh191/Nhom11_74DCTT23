import pandas as pd
import numpy as np
import pickle
from lightfm import LightFM
from scipy.sparse import load_npz, csr_matrix
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

# === Gợi ý theo Customer_ID ===
def recommend_products(customer_id, top_n=5):
    customer_id = str(customer_id).strip()

    if customer_id not in allowed_ids:
        return f"❌ Customer_ID '{customer_id}' không có trong dữ liệu huấn luyện.", ""

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

# === Top sản phẩm được đánh giá nhiều nhất ===
def most_rated_products(top_n=5):
    item_counts = df_train["Item_Purchased"].value_counts().head(top_n)
    top_items = item_counts.index.tolist()
    result_html = render_item_boxes(top_items)
    return f"📦 Top {top_n} sản phẩm được nhiều người đánh giá nhất:", result_html

# === Gợi ý cho người dùng mới, ƯU TIÊN CHÍNH XÁC CATEGORY ===
def recommend_for_new_user(age, gender, category, season, top_n):
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

    age_group_val = age_group(age)
    feature_names = [f"Gender={gender}", f"Age_Group={age_group_val}"]
    feature_index_map = dataset._user_feature_mapping

    indices = []
    values = []
    for feat in feature_names:
        if feat in feature_index_map:
            indices.append(feature_index_map[feat])
            values.append(1.0)

    new_user_vec = csr_matrix((values, ([0]*len(indices), indices)),
                              shape=(1, user_features.shape[1]))

    df_items = df_train.drop_duplicates("Item_Purchased")[["Item_Purchased", "Category", "Season"]]
    filtered_items = df_items[df_items["Category"] == category]

    if filtered_items.empty:
        return f"❌ Không tìm thấy sản phẩm nào trong danh mục '{category}'", ""

    item_id_to_index = {v: k for k, v in item_id_reverse.items()}
    filtered_indexes = [
        item_id_to_index[item_id]
        for item_id in filtered_items["Item_Purchased"]
        if item_id in item_id_to_index
    ]

    if not filtered_indexes:
        return f"❌ Không tìm thấy mã sản phẩm phù hợp với mô hình.", ""

    scores = model.predict(0, filtered_indexes,
                           user_features=new_user_vec,
                           item_features=item_features)

    top_indices = np.argsort(-scores)[:top_n]
    top_items = [item_id_reverse[filtered_indexes[i]] for i in top_indices]

    return render_item_boxes(top_items)

# === Giao diện Gradio với Tabs ===
with gr.Blocks(title="Hệ thống gợi ý sản phẩm") as demo:
    with gr.Tabs():
        with gr.Tab("🎯 Gợi ý theo Customer_ID"):
            gr.Markdown("## 🎯 Gợi ý sản phẩm dựa trên Customer_ID")
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

        with gr.Tab("🔥 Sản phẩm phổ biến"):
            gr.Markdown("## 🔥 Top sản phẩm được nhiều người đánh giá nhất")
            with gr.Row():
                topn_popular = gr.Slider(1, 20, value=5, step=1, label="Số sản phẩm hiển thị")
                btn2 = gr.Button("📊 Xem danh sách")
            output_text2 = gr.Textbox(label="Thông tin", interactive=False)
            output_html2 = gr.HTML()
            btn2.click(fn=most_rated_products,
                       inputs=topn_popular,
                       outputs=[output_text2, output_html2])

        with gr.Tab("🆕 Gợi ý cho người dùng mới"):
            gr.Markdown("## 🆕 Gợi ý sản phẩm cho người dùng mới")
            with gr.Row():
                with gr.Column():
                    age_input = gr.Number(label="Tuổi", value=25)
                    gender_input = gr.Dropdown(choices=["Male", "Female"], label="Giới tính")
                    category_input = gr.Dropdown(choices=["Clothing", "Accessories", "Outerwear", "Footwear"], label="Danh mục ưa thích")
                    season_input = gr.Dropdown(choices=["Spring", "Summer", "Fall", "Winter"], label="Mùa yêu thích")
                    topn_newuser = gr.Slider(1, 20, value=5, step=1, label="Số sản phẩm gợi ý")
                    btn3 = gr.Button("✨ Gợi ý ngay")
                result_newuser = gr.HTML()

            btn3.click(fn=recommend_for_new_user,
                       inputs=[age_input, gender_input, category_input, season_input, topn_newuser],
                       outputs=result_newuser)

# === Chạy Gradio ===
demo.launch()
