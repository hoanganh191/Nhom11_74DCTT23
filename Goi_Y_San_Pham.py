import pickle
import numpy as np
import pandas as pd
from lightfm import LightFM

# ===== Hàm hỗ trợ =====
def load_npz_object(path):
    return np.load(path, allow_pickle=True)["data"].item()

# ===== 1. Load mô hình =====
with open("MODEL/lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

# ===== 2. Load các ma trận =====
train_interactions = load_npz_object("MODEL/train_interactions.npz")
val_interactions = load_npz_object("MODEL/val_interactions.npz")
test_interactions = load_npz_object("MODEL/test_interactions.npz")

user_features_matrix = load_npz_object("MODEL/user_features_matrix.npz")
item_features_matrix = load_npz_object("MODEL/item_features_matrix.npz")


