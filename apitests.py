import requests

print(requests.post("http://localhost:8000/Kinematics/invoke", json={'input': {'a': 'How would I solve a question asking me to find the distance travelled by a ball moving at 10m/s for one minute?'}}).json())

