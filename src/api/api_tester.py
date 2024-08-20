import requests

url = 'http://127.0.0.1:8000/predict'
data = {
    'credit_score': 600,
    'geography': 'Germany',
    'gender': 'Male',
    'age': 30,
    'tenure': 5,
    'balance': 50000.0,
    'num_of_products': 2,
    'has_cr_card': 1,
    'is_active_member': 1,
    'estimated_salary': 70000.0
}

response = requests.post(url, json=data)

print(f'Status Code: {response.status_code}')
print(f'Response Text: {response.text}')

try:
    response_json = response.json()
    print(response_json)
except requests.exceptions.JSONDecodeError:
    print("Erro ao tentar decodificar a resposta como JSON")