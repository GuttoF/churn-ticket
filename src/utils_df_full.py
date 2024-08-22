from pathlib import Path
import pandas as pd

# found the main project folders
path = Path().resolve().parent
data_path = path / "data/processed"

X_train = pd.read_parquet(data_path / "X_train_fs.parquet")
X_test = pd.read_parquet(data_path / "X_test_fs.parquet")
X_val = pd.read_parquet(data_path / "X_val_fs.parquet")
y_train = pd.read_pickle(data_path / "y_train.pkl")
y_test = pd.read_pickle(data_path / "y_test.pkl")
y_val = pd.read_pickle(data_path / "y_val.pkl")

y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)
y_val_df = pd.DataFrame(y_val)

train_data = pd.concat([X_train, y_train_df], axis=1)
test_data = pd.concat([X_test, y_test_df], axis=1)
val_data = pd.concat([X_val, y_val_df], axis=1)

df = pd.concat([train_data, test_data, val_data], axis=0)
df.to_parquet(data_path / "df.parquet")