import requests

api_url = os.getenv("API_URL")
#api_url = "http://127.0.0.1:8000/predict"
data = {
    "credit_score": 0,
    "geography": "Germany",
    "gender": "Male",
    "age": 30,
    "tenure": 0,
    "balance": 1,
    "num_of_products": 0,
    "has_cr_card": 1,
    "is_active_member": 1,
    "estimated_salary": 0,
}

response = requests.post(api_url, json=data)

print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

try:
    response_json = response.json()
    print(response_json)
except requests.exceptions.JSONDecodeError:
    print("Erro ao tentar decodificar a resposta como JSON")
