# =====================================================
# DATA PREPARATION FOR MACHINE LEARNING MODELS
# Diploma Thesis – Reproducible Version
# =====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from google.colab import files
import io

print("=== DATA PREPARATION FOR MODELING ===")
print("Upload dataset (Excel or CSV)\n")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

if file_name.endswith(('.xlsx', '.xls')):
    df = pd.read_excel(io.BytesIO(uploaded[file_name]))
elif file_name.endswith(".csv"):
    df = pd.read_csv(io.BytesIO(uploaded[file_name]))
else:
    raise Exception("Unsupported file format")

print("Original sample size:", len(df))

# =====================================================
# CLEANING
# =====================================================

numeric_cols = ['C', 'H', 'N', 'S', 'O', 'HHV (Mj kg)']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['C', 'H', 'N', 'S', 'O', 'HHV (Mj kg)', 'Kategorija'])
df = df.drop_duplicates()

print("Final sample size after cleaning:", len(df))

# =====================================================
# DEFINE INPUTS AND OUTPUT
# =====================================================

X = df[['C', 'H', 'N', 'S', 'O', 'Kategorija']]
y = df['HHV (Mj kg)']

numeric_features = ['C', 'H', 'N', 'S', 'O']
categorical_features = ['Kategorija']

# =====================================================
# DATA SPLITTING (70 / 15 / 15)
# =====================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# =====================================================
# SAVE RAW SPLITS
# =====================================================

train_raw = X_train.copy()
train_raw['HHV (Mj kg)'] = y_train.values

val_raw = X_val.copy()
val_raw['HHV (Mj kg)'] = y_val.values

test_raw = X_test.copy()
test_raw['HHV (Mj kg)'] = y_test.values

train_raw.to_csv("train_raw.csv", index=False)
val_raw.to_csv("validation_raw.csv", index=False)
test_raw.to_csv("test_raw.csv", index=False)

# =====================================================
# PREPROCESSING (SCALING + ONE-HOT ENCODING)
# =====================================================

numeric_transformer = StandardScaler()

# --- version-safe OneHotEncoder ---
try:
    categorical_transformer = OneHotEncoder(
        drop='first',
        sparse_output=False
    )
except TypeError:
    categorical_transformer = OneHotEncoder(
        drop='first',
        sparse=False
    )

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit ONLY on training data
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

# =====================================================
# SAVE PREPROCESSED DATA
# =====================================================

feature_names_num = numeric_features
feature_names_cat = list(
    preprocessor.named_transformers_['cat']
    .get_feature_names_out(categorical_features)
)

feature_names = feature_names_num + feature_names_cat

X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
X_val_df = pd.DataFrame(X_val_proc, columns=feature_names)
X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

X_train_df['HHV (Mj kg)'] = y_train.values
X_val_df['HHV (Mj kg)'] = y_val.values
X_test_df['HHV (Mj kg)'] = y_test.values

X_train_df.to_csv("train_preprocessed.csv", index=False)
X_val_df.to_csv("validation_preprocessed.csv", index=False)
X_test_df.to_csv("test_preprocessed.csv", index=False)

# =====================================================
# DOWNLOAD FILES
# =====================================================

files.download("train_raw.csv")
files.download("validation_raw.csv")
files.download("test_raw.csv")

files.download("train_preprocessed.csv")
files.download("validation_preprocessed.csv")
files.download("test_preprocessed.csv")

print("\nAll datasets prepared and downloaded successfully.")
