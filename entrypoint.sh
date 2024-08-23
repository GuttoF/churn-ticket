#!/bin/sh

if ! [ -z "$PORT" ] && [ "$PORT" -eq "$PORT" ] 2>/dev/null; then
    exec poetry run streamlit run src/app/main.py --server.port=$PORT
else
    echo "PORT variable is not set or not a valid integer"
    exit 1
fi
