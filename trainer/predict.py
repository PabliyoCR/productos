import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

EXCEL_PATH = "../../DatosArticulos/productos_test.xlsx"
IMAGES_DIR = "../../DatosArticulos/images"
MODEL_DIR = "./clip_finetuned" # Actualizado para coincidir con OUTPUT_DIR de train.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Cargar modelo y data ====
processor = CLIPProcessor.from_pretrained(MODEL_DIR)
model = CLIPModel.from_pretrained(MODEL_DIR).to(DEVICE)

df = pd.read_excel(EXCEL_PATH).dropna(subset=["Nombre"])

# Todos los textos de productos, construidos igual que en el entrenamiento
productos = []
for index, row in df.iterrows():
    text = f"{row['No_articulo']} | {row['Nombre']} | {row['Familia']} | {row['Categoria']} | {row['Subcategoria']}"
    productos.append(text)

text_inputs = processor(text=productos, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
with torch.no_grad():
    text_embeds = model.get_text_features(**text_inputs)
text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True)

# ==== Predicción ====
def predecir(img_path, top_k=5):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_embed = model.get_image_features(**inputs)
    image_embed /= image_embed.norm(p=2, dim=-1, keepdim=True)

    sims = (image_embed @ text_embeds.T).squeeze(0)
    topk = sims.topk(top_k)
    resultados = [(productos[i], float(sims[i])) for i in topk.indices]
    return resultados

if __name__ == "__main__":
    ejemplo = "./image.png"
    print(predecir(ejemplo))
