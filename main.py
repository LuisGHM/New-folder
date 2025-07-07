from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import requests
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mude para domínios seguros depois
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageInput(BaseModel):
    image_url: str

@app.post("/classify")
async def classify(image: ImageInput):
    if not image.image_url:
        raise HTTPException(status_code=400, detail="Erro: Nenhuma entrada foi fornecida.")

    try:
        result = analyze_image(image.image_url)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def analyze_image(image_url: str):
    img = get_image_from_url(image_url)
    if img is None:
        raise HTTPException(status_code=400, detail="Erro ao obter a imagem.")

    # Diretório base e paths fixos dos modelos
    BASE_DIR = os.path.abspath(os.getcwd())
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    category_model_path = os.path.join(MODEL_DIR, "classificationCategory.pt")
    clothes_model_path = os.path.join(MODEL_DIR, "classificationClothes.pt")

    # Verifica existência dos arquivos
    if not os.path.isfile(category_model_path) or not os.path.isfile(clothes_model_path):
        raise HTTPException(status_code=500, detail="Erro: Modelos classificationCategory.pt ou classificationClothes.pt não encontrados em /models.")

    # Carrega os dois modelos (ambos agora são de classificação)
    try:
        category_model = YOLO(category_model_path)
        clothes_model = YOLO(clothes_model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar os modelos: {e}")

    # Classifica com cada modelo
    category_name, category_conf = classify_image(img, category_model)
    clothes_name, clothes_conf = classify_image(img, clothes_model)

    return {
        "classificationCategory": {"class_name": category_name, "confidence": category_conf},
        "classificationClothes": {"class_name": clothes_name, "confidence": clothes_conf},
    }


def get_image_from_url(url: str):
    try:
        if url.startswith("data:image"):
            _, encoded = url.split(",", 1)
            data = base64.b64decode(encoded)
            img = Image.open(BytesIO(data)).convert("RGB")
        else:
            resp = requests.get(url)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Erro ao carregar a imagem da URL {url}: {e}")
        return None


def classify_image(img: Image.Image, model: YOLO):
    # Redimensiona e converte para array
    img_resized = img.resize((640, 640))
    img_array = np.array(img_resized)

    # Predição (classificação)
    results = model.predict(source=img_array, verbose=False, save=False)
    if results:
        probs = results[0].probs
        top1 = probs.top1
        class_name = model.names[top1]
        confidence = float(probs.top1conf.item())
        return class_name, confidence

    return None, None
