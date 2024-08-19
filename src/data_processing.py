import os
import re
import sys
from pathlib import Path
import duckdb


def snake_case(column_name: str) -> str:
    """
    Converts a CamelCase or PascalCase string to snake_case.

    Args:
    - column_name: A string representing the column name.

    Returns:
    - str: The column name in snake_case.
    """
    name_with_underscores = re.sub(r"(?<!^)(?=[A-Z])", "_", column_name)

    return name_with_underscores.lower()


main_path = Path().resolve().parent
# path = os.path.join(main_path, "churn-ticket")

full_path = os.path.join(main_path, "data/raw/churn.csv")
new_path = os.path.join(main_path, "data/interim/churn.db")

# verify if the file exists
if not os.path.exists(full_path):
    print(f"File {full_path} not found.")
    sys.exit()

if os.path.exists(new_path):
    print(f"The file located in {new_path} already exists.")
    sys.exit()

try:
    conn = duckdb.connect(new_path)
    conn.execute(f"CREATE TABLE churn AS SELECT * FROM read_csv_auto('{full_path}')")

    # rename columns to snake case
    columns_info = conn.execute("PRAGMA table_info('churn')").fetchall()
    for column in columns_info:
        old_name = column[1]
        new_name = snake_case(old_name)
        conn.execute(f"ALTER TABLE churn RENAME COLUMN {old_name} TO {new_name}")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit()
finally:
    conn.close()
