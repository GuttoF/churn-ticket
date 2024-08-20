import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv
import os
import logging


def get_environment_choice():
    print("Select the environment to upload:")
    print("1. Production")
    print("2. Development")

    while True:
        try:
            aux = int(input("Enter your choice (1 or 2): "))
            if aux in [1, 2]:
                break
            else:
                print("Invalid choice, please select 1 or 2.")
        except ValueError:
            print("Invalid input, please enter a number (1 or 2).")

    return aux


# Path to .env
path = Path().resolve().parent
data_path = path / 'data/interim/churn.csv'
env_path = path / '.env'

load_dotenv(dotenv_path=env_path)

result = get_environment_choice()

if result == 1:
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    dbname = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
else:
    host = os.getenv("PGHOST_H")
    port = os.getenv("PGPORT_H")
    dbname = os.getenv("PGDATABASE_H")
    user = os.getenv("PGUSER_H")
    password = os.getenv("PGPASSWORD_H")

# Connect to the database
engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')

# Load CSV file
df = pd.read_csv(data_path)

# Send data to the database
try:
    connection = engine.connect()
    df.to_sql('churn', engine, if_exists='replace', index=False)
    logging.info('Data uploaded to the database')
    connection.close()
except Exception as e:
    logging.error(f'Error uploading data to the database: {e}')




