import cudf
import cupy as cp
import pickle
import os
from pathlib import Path

# Đảm bảo thư mục đầu ra tồn tại
output_dir = "data/processed"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Đọc dữ liệu
train_file = "data/raw/yoochoose-clicks.dat"
test_file = "data/test/yoochoose-test.dat"

# Kiểm tra tệp đầu vào tồn tại
if not os.path.exists(train_file):
    raise FileNotFoundError(f"Train file {train_file} does not exist.")
if not os.path.exists(test_file):
    raise FileNotFoundError(f"Test file {test_file} does not exist.")

try:
    train_gdf = cudf.read_csv(
        train_file,
        header=None,
        names=["session_id", "timestamp", "item_id", "category"],
    )
    test_gdf = cudf.read_csv(
        test_file, header=None, names=["session_id", "timestamp", "item_id", "category"]
    )
except Exception as e:
    raise RuntimeError(f"Error reading input files: {str(e)}")

# Sort by session_id and time
train_gdf = train_gdf.sort_values(["session_id", "timestamp"])
test_gdf = test_gdf.sort_values(["session_id", "timestamp"])

# Encode item_id
unique_items = train_gdf["item_id"].unique()
item_encoder = cudf.Series(cp.arange(1, len(unique_items) + 1), index=unique_items)
train_gdf["item_idx"] = train_gdf["item_id"].map(item_encoder)
test_gdf["item_idx"] = test_gdf["item_id"].map(item_encoder)

# Xử lý dữ liệu test: loại bỏ các hàng không có item_idx và ép kiểu
test_gdf = test_gdf.dropna(subset=["item_idx"])
train_gdf["item_idx"] = train_gdf["item_idx"].astype("int32")
test_gdf["item_idx"] = test_gdf["item_idx"].astype("int32")

# Lưu item_encoder
try:
    with open(f"{output_dir}/item_encoder.pkl", "wb") as f:
        pickle.dump(item_encoder.to_pandas(), f)
except Exception as e:
    raise RuntimeError(f"Error saving item_encoder.pkl: {str(e)}")

# Lưu train_gdf và test_gdf
try:
    train_gdf.to_parquet(f"{output_dir}/clicks_train.parquet")
    test_gdf.to_parquet(f"{output_dir}/clicks_test.parquet")
except Exception as e:
    raise RuntimeError(f"Error saving Parquet files: {str(e)}")

# Group by session_id to create sequences
sessions_gdf = train_gdf.groupby("session_id")["item_idx"].agg("collect")
sessions = sessions_gdf.to_arrow().to_pylist()
test_sessions_gdf = test_gdf.groupby("session_id")["item_idx"].agg("collect")
test_sessions = test_sessions_gdf.to_arrow().to_pylist()

# Lưu sessions và test_sessions
try:
    with open(f"{output_dir}/train_sessions.pkl", "wb") as f:
        pickle.dump(sessions, f)
    with open(f"{output_dir}/test_sessions.pkl", "wb") as f:
        pickle.dump(test_sessions, f)
except Exception as e:
    raise RuntimeError(f"Error saving session files: {str(e)}")

# In thông tin
print(f"Number of unique items: {len(unique_items)}")
print(f"Number of sessions: {len(sessions)}")
print(f"Number of test sessions: {len(test_sessions)}")
