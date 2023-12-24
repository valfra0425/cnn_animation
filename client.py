import requests
import json
from PIL import Image

# Open image and convert to bytes
image = Image.open("boxe.jpeg")
image = image.convert("RGBA")
image_bytes = image.tobytes()

response = requests.post(
    "http://127.0.0.1:5000/predict",
    headers={"Content-Type": "application/json"},
    data=image_bytes,
)

if response.status_code == 200:
    response_content = response.content.decode()
    response_data = json.loads(response_content)
    print(response_data)
else:
    print("Error: {}".format(response.status_code))
