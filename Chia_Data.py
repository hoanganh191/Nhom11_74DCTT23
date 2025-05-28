import pandas as pd
from sklearn.model_selection import train_test_split

# Bước 0: Đọc dữ liệu
df = pd.read_csv("/content/Data_XuLy_Chot.csv")

# Bước 1: Lấy danh sách người dùng duy nhất
unique_users = df['Customer_ID'].unique()

# Bước 2: Chia người dùng thành 80% train, 20% còn lại để chia val và test
train_users, eval_users = train_test_split(unique_users, test_size=0.2, random_state=42)

# Bước 3: Chia 20% còn lại thành 10% val và 10% test
val_users, test_users_raw = train_test_split(eval_users, test_size=0.5, random_state=42)

# Bước 4: Phân chia warm/cold cho val và test
warm_val_users = [u for u in val_users if u in train_users]
cold_val_users = [u for u in val_users if u not in train_users]

warm_test_users = [u for u in test_users_raw if u in train_users]
cold_test_users = [u for u in test_users_raw if u not in train_users]

# Bước 5: Tạo tập train
df_train = df[df['Customer_ID'].isin(train_users)].copy()
df_train['Phase'] = 'train'
df_train['User_Type'] = 'warm'  # train mặc định là warm

# Bước 6: Tạo tập validation (val)
df_val_warm = df[df['Customer_ID'].isin(warm_val_users)].copy()
df_val_warm['Phase'] = 'val'
df_val_warm['User_Type'] = 'warm'

df_val_cold = df[df['Customer_ID'].isin(cold_val_users)].copy()
df_val_cold['Phase'] = 'val'
df_val_cold['User_Type'] = 'cold'

df_val = pd.concat([df_val_warm, df_val_cold], ignore_index=True)

# Bước 7: Tạo tập test
df_test_warm = df[df['Customer_ID'].isin(warm_test_users)].copy()
df_test_warm['Phase'] = 'test'
df_test_warm['User_Type'] = 'warm'

df_test_cold = df[df['Customer_ID'].isin(cold_test_users)].copy()
df_test_cold['Phase'] = 'test'
df_test_cold['User_Type'] = 'cold'

df_test = pd.concat([df_test_warm, df_test_cold], ignore_index=True)

# Bước 8: Lưu thành các file CSV
df_train.to_csv("data_train.csv", index=False)
df_val.to_csv("data_val.csv", index=False)
df_test.to_csv("data_test.csv", index=False)

# In thống kê
print("✅ Đã chia và lưu các tập dữ liệu thành công.")
print(f"Số dòng train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
print(f"Người dùng val (warm): {len(warm_val_users)}, (cold): {len(cold_val_users)}")
print(f"Người dùng test (warm): {len(warm_test_users)}, (cold): {len(cold_test_users)}")
