import numpy as np
import os
import pandas as pd
from csv import reader

#BASE_PATH = "/content/drive/MyDrive/Bha Project/146203bk2/Main/Processed"
BASE_PATH = "/content/drive/MyDrive/Bha Praject-2(jaanu)/Bha Project/146203bk2/Main/Processed"
def load_csv(filename):
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        return [row for row in csv_reader if row]

def convert(data):
    # Replace 0.0 with 0.1 to avoid zero values if needed
    return [[0.1 if val == 0.0 else float(val) for val in row] for row in data]

def find_class(Z, dts):
    col = Z[:, -1]
    
    if dts == 'Adult' or dts == 'adult_UCI':
        return [0 if c.strip() in ('<=50K', ' <=50K') else 1 for c in col]
    
    elif dts == 'Credit_Approval':
        return [0 if c.strip() == '-' else 1 for c in col]
    
    elif dts == 'creditcard':
        return [int(c) for c in col]
    
    # elif dts == 'kddcup99_cleaned':
    #     labels = sorted(set(Z[:, -1]))  # sorted for consistent indexing
    #     return [labels.index(c) for c in Z[:, -1]]
    
    elif dts == '7817_1_cleaned':
        # Assuming label is last column as float ratings
        rating_col_index = -1  
        labels = []
        for val in Z[:, rating_col_index]:
            try:
                if val.strip() != '':
                    labels.append(float(val))
                else:
                    labels.append(0.0)
            except Exception:
                labels.append(0.0)  # fallback for bad data
        return labels
    
    else:
        return [0 for _ in col]

def find_unique(column):
    return [val for val in np.unique(column) if val.strip() not in ('?', '')]

def str_convert(column, unique_values):
    return [unique_values.index(val) if val.strip() not in ('?', '') else -1 for val in column]

def find_missing(data):
    # Convert values to float safely, replace non-convertible or '?' with np.nan
    converted = []
    for row in data:
        new_row = []
        for val in row:
            try:
                new_row.append(float(val) if val != '?' else np.nan)
            except Exception:
                new_row.append(np.nan)
        converted.append(new_row)
    data = np.array(converted, dtype=np.float32)

    # Replace NaNs with column means
    if np.isnan(data).sum() > 0:
        col_means = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_means, inds[1])
    return data

def preprocess_7817_1_cleaned(data):
    # Example: parse 'weight' column (assumed index 26) to float grams
    weight_col = 26
    for i, val in enumerate(data[:, weight_col]):
        if isinstance(val, str) and 'grams' in val:
            try:
                data[i, weight_col] = float(val.replace('grams', '').strip())
            except Exception:
                data[i, weight_col] = 0.0
        else:
            try:
                data[i, weight_col] = float(val)
            except Exception:
                data[i, weight_col] = 0.0
    # Convert to float type explicitly
    data[:, weight_col] = data[:, weight_col].astype(float)

    # TODO: Add any other preprocessing (e.g., dimensions parsing) here if needed
    
    return data

def process_dataset(dts):
    filename = f"/content/drive/MyDrive/Bha Praject-2(jaanu)/Bha Project/146203bk2/Main/Data/{dts}.csv"
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")

    raw_data = load_csv(filename)[1:]  # Skip header
    data = np.array(raw_data)

    # Special preprocessing for 7817_1_cleaned
    if dts == '7817_1_cleaned':
        data = preprocess_7817_1_cleaned(data)

    data = data.T

    categorical_map = {
        'Adult': [1, 3, 5, 6, 7, 8, 9, 13],
        'adult_UCI': [1, 3, 5, 6, 7, 8, 9, 13],
        'Credit_Approval': [0, 3, 4, 5, 6, 8, 9, 11, 12],
        # 'kddcup99_cleaned': [1, 2, 3],
        '7817_1_cleaned': [2, 3, 4, 10],  # adjust if needed
    }

    # Encode categorical columns to numeric indices
    for idx in categorical_map.get(dts, []):
        unique_vals = find_unique(data[idx])
        data[idx] = str_convert(data[idx], unique_vals)

    data = data.T

    if dts == 'transfers_sorted':
        # Example special handling for transfers_sorted dataset
        numeric_cols = []
        for i in range(data.shape[1]):
            try:
                float(data[0][i])
                numeric_cols.append(i)
            except Exception:
                continue
        X = data[:, numeric_cols]
        Y = [0 for _ in range(len(X))]  # Dummy labels for this dataset
    
    else:
        X, Y = data[:, :-1], find_class(data, dts)

    X = find_missing(X)

    # Save processed data and labels
    np.savetxt(os.path.join(BASE_PATH, f"{dts}.csv"), X, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(BASE_PATH, f"{dts}_label.csv"), Y, delimiter=',', fmt='%s')

    return convert(X)

def read_input(dts):
    data_path = os.path.join(BASE_PATH, f"{dts}.csv")
    label_path = os.path.join(BASE_PATH, f"{dts}_label.csv")

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        print(f"🔄 Processed files not found for '{dts}'. Generating...")
        process_dataset(dts)
        print("✅ Files generated successfully.")

    data = pd.read_csv(data_path, header=None).values
    label = pd.read_csv(label_path, header=None).values
    
    return data, label
