from PIL import Image
from flask import Flask, request, jsonify
from joblib import load
import wandb
import os
from dotenv import load_dotenv
from keras.models import load_model
import tensorflow as tf
import platform

load_dotenv()  # Load variables from .env file

wandb_api_key = os.getenv("WANDB_API_KEY")

model = None

if wandb_api_key:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
else:
    print("WANDB_API_KEY not found. Please set the environment variable.")

api = wandb.Api()
artifact = api.artifact('flamigos/cnn_animation/animation_model_pipeline:latest', type='pipeline')
arquivo = artifact.file()
artifact2 = api.artifact('flamigos/cnn_animation/imageProcessor_class:latest', type='python')
artifact_dir = artifact2.file()

dest_path = './'

if platform.system() == "Windows":
    os.system(f'copy "{artifact_dir}" "{dest_path}"')
else:
    os.system(f'cp {artifact_dir} {dest_path}')

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    image_bytes = request.data
    image = Image.frombytes("RGBA", (300, 300), image_bytes)

    p = model.predict([image])
    if p[0][0] == 0:
        result = "cartoon"
    else:
        result = "anime"
    print("Predição realizada com sucesso!")
    print(result)
    return jsonify({'message': result})


@app.route("/")
def index():
    return '''<h1>Bem vindo ao servidor de Predição</h1> <p>Aqui você pode predizer imagens de animações entre cartoon
     e anime </p>'''


if __name__ == "__main__":
    model = load(arquivo)
    app.run(debug=True)
