import pandas as pd
from sklearn.model_selection import train_test_split

# Bước 0: Đọc dữ liệu
df = pd.read_csv("/content/Data_Dong_Bo.csv")

# Bước 1: Lấy danh sách người dùng duy nhất
unique_users = df['Customer_ID'].unique()

# Bước 2: Chia 70% train, 30% còn lại để chia val và test
train_users, remaining_users = train_test_split(unique_users, test_size=0.3, random_state=42)

# Bước 3: Chia 30% còn lại thành 10% val (cold), 20% còn lại để chia test warm và test cold
val_users, test_cold_users = train_test_split(remaining_users, test_size=2/3, random_state=42)

# Bước 4: Chọn test warm users từ train users
test_warm_users, _ = train_test_split(train_users, test_size=0.1 / 0.7, random_state=42)
# Giải thích: vì train chiếm 70%, nên cần lấy đúng 10% tổng số user → 0.1/0.7 ~ 14.3% trong tập train

# Bước 5: Tạo tập train
df_train = df[df['Customer_ID'].isin(train_users)].copy()

# Bước 6: Tạo tập validation cold
df_val = df[df['Customer_ID'].isin(val_users)].copy()

# Bước 7: Tạo tập test warm
df_test_warm = df[df['Customer_ID'].isin(test_warm_users)].copy()

# Bước 8: Tạo tập test cold
df_test_cold = df[df['Customer_ID'].isin(test_cold_users)].copy()

# Bước 9: Lưu thành các file CSV riêng biệt
df_train.to_csv("data_train.csv", index=False)
df_val.to_csv("data_val_cold.csv", index=False)
df_test_warm.to_csv("data_test_warm.csv", index=False)
df_test_cold.to_csv("data_test_cold.csv", index=False)

# In thống kê
print("✅ Đã chia và lưu các tập dữ liệu thành công.")
print(f"Số dòng train: {len(df_train)}")
print(f"Số dòng val (cold): {len(df_val)}")
print(f"Số dòng test warm: {len(df_test_warm)}")
print(f"Số dòng test cold: {len(df_test_cold)}")
print(f"Số người dùng train: {len(train_users)}")
print(f"Số người dùng val (cold): {len(val_users)}")
print(f"Số người dùng test warm: {len(test_warm_users)}")
print(f"Số người dùng test cold: {len(test_cold_users)}")
