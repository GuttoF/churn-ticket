#!/bin/sh

exec poetry run streamlit run src/app/main.py --server.port="$PORT"