import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'experience':2, 'test_score(out of 10)':9, 'interview_score(out of 10)':6})

print(r.json())