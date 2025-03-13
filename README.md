# Bank Customer Churn Prediction Project

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Running the Project](#running-the-project)
- [API Documentation](#api-documentation)
- [Web Application Features](#web-application-features)
- [Data Processing and Model Training](#data-processing-and-model-training)
- [Testing](#testing)
- [License](#license)

## Overview

This project implements a comprehensive churn prediction system for bank customers. It consists of a machine learning API for making predictions and a web-based user interface for visualizing analytics and interacting with the prediction model.

## Project Structure

```bash
churn-ticket/
├── data/
│   ├── interim/
│   │   └── churn.db
│   └── processed/
│       ├── df.parquet
│       ├── Xtestfs.parquet
│       ├── Xtrainfs.parquet
│       ├── y_test.pkl
│       └── y_train.pkl
├── docs/
├── notebooks/
├── src/
│   ├── api/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── services.py
│   ├── app/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── page_topics/
│   ├── models/
│   └── scalers/
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Key Components

### Machine Learning API (`src/api`)

- Built with FastAPI
- Provides endpoints for churn prediction
- Implements feature engineering and machine learning model

### Web Application (`src/app`)

- Developed with Streamlit
- Offers interactive data visualizations and analytics
- Allows users to make individual churn predictions

### Data Analysis and Model (`src`)

- Scripts for data processing, feature engineering, and model training
- Utilities for visualization and model evaluation

## Technologies Used

- Python 3.12
- FastAPI
- Streamlit
- Pandas, NumPy, Scikit-learn
- CatBoost
- Docker and Docker Compose
- Poetry for dependency management

## Setup and Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/churn-ticket.git
   cd churn-ticket
   ```

2. Install dependencies (for local development):

   ```sh
   poetry install
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add necessary environment variables (e.g., `APIURL`, `DATABASEURL`)

## Running the Project

### Using Docker Compose (recommended for production)

```sh
docker-compose up --build
```

### Running locally (for development)

```sh
# Start the API
poetry run uvicorn src.api.main:app --reload

# Start the Streamlit app
poetry run streamlit run src/app/main.py
```

### Accessing the applications

- **API**: [http://localhost:8000](http://localhost:8000)
- **Web Interface**: [http://localhost:8501](http://localhost:8501)

## API Documentation

- Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)

### Key endpoints

- `/predict`: `POST` request for churn prediction

## Web Application Features

- **Home**: Project overview and data dictionary
- **Model Explanation**: Details on the machine learning model used
- **Exploratory Data Analysis**: Interactive visualizations of customer data
- **Top Clients**: Analysis of high-value customers
- **Churn Prediction**: Interface for making individual predictions
- **Ticket Simulation**: Tool for simulating customer retention strategies
- **About**: Information about the project and developer

## Data Processing and Model Training

- Data preprocessing scripts are located in `src/data_processing.py`
- Model training script is in `src/model_training.py`
- Feature engineering utilities are in `src/utils_feature_engineering.py`

## Testing

Run tests using `pytest`:

```sh
poetry run pytest
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
