# preprocess.py
import os
import time
import cudf
import cupy as cp
import pickle

def load_and_preprocess_data(file_path):
    # Define data folder and file paths
    train_file = os.path.join(file_path, "yoochoose/yoochoose-clicks.dat")
    test_file = os.path.join(file_path, "yoochoose/yoochoose-test.dat")

    # Load data with cudf
    start_time = time.time()
    train_gdf = cudf.read_csv(
        train_file,
        names=['session_id', 'time', 'item_id', 'category'],
        dtype={'session_id': 'int32', 'item_id': 'int32', 'category': 'str'},
        parse_dates=['time']
    )
    end_time = time.time()
    print(f"cuDF Load Time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    test_gdf = cudf.read_csv(
        test_file,
        names=['session_id', 'time', 'item_id', 'category'],
        dtype={'session_id': 'int32', 'item_id': 'int32', 'category': 'str'},
        parse_dates=['time']
    )
    end_time = time.time()
    print(f"cuDF Test Load Time: {end_time - start_time:.4f} seconds")

    # Sort by session_id and time
    train_gdf = train_gdf.sort_values(['session_id', 'time'])
    test_gdf = test_gdf.sort_values(['session_id', 'time'])

    # Encode item_id using cudf and cupy
    unique_items = train_gdf['item_id'].unique()
    item_encoder = cudf.Series(cp.arange(1, len(unique_items) + 1), index=unique_items)
    train_gdf['item_idx'] = train_gdf['item_id'].map(item_encoder)
    test_gdf['item_idx'] = test_gdf['item_id'].map(item_encoder)
    test_gdf = test_gdf.dropna(subset=['item_idx'])
    test_gdf['item_idx'] = test_gdf['item_idx'].astype('int32')

    # Group by session_id to create sequences
    sessions_gdf = train_gdf.groupby('session_id')['item_idx'].agg('collect')
    sessions = sessions_gdf.to_arrow().to_pylist()

    test_sessions_gdf = test_gdf.groupby('session_id')['item_idx'].agg('collect')
    test_sessions = test_sessions_gdf.to_arrow().to_pylist()

    # Print statistics
    print(f"Number of unique items: {len(unique_items)}")
    print(f"Number of sessions: {len(sessions)}")
    print(f"Number of test sessions: {len(test_sessions)}")

    # Save processed data
    save_dir = '/mnt/c/Users/HP/mlops_cs317/data/processed/'
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir + 'item_encoder.pkl', 'wb') as f:
        pickle.dump(item_encoder.to_pandas(), f)

    train_gdf.to_parquet(save_dir + 'clicks_train.parquet', index=False)
    test_gdf.to_parquet(save_dir + 'clicks_test.parquet', index=False)

    with open(save_dir + 'train_sessions.pkl', 'wb') as f:
        pickle.dump(sessions, f)

    with open(save_dir + 'test_sessions.pkl', 'wb') as f:
        pickle.dump(test_sessions, f)

    print("Saved all successfully.")

    return sessions, test_sessions, len(unique_items)