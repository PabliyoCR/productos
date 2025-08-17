# server.py
import faiss
import torch
import clip
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tempfile
from flask_cors import CORS

INDEX_FILE = "productos.index"
MAP_FILE = "mapping.npy"

app = Flask(__name__)
CORS(app)  # permite peticiones desde React

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# Cargar modelo e índice solo una vez
model, preprocess, device = load_model()
if device == "cuda":
    print("Cargando índice FAISS en GPU...")
    index = faiss.read_index(INDEX_FILE)
    index = faiss.index_gpu_to_cpu(index) # Mover a CPU para evitar problemas de compatibilidad o si solo la búsqueda se hace en GPU
    # Para usar FAISS en GPU directamente, necesitarías un IndexFlatGPU o similar
    # y transferir el índice a la GPU después de cargarlo.
    # Por simplicidad, lo cargamos en CPU y luego podríamos moverlo si fuera necesario.
else:
    print("Cargando índice FAISS en CPU...")
    index = faiss.read_index(INDEX_FILE)
mapping = np.load(MAP_FILE, allow_pickle=True)

# Considerar usar un modelo CLIP más pequeño (ej. "ViT-B/16" o un modelo cuantificado)
    # si los problemas de memoria persisten, o si se despliega en un entorno con recursos limitados.
def buscar_producto(imagen_path, top_k=100):
    img = Image.open(imagen_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy()

    D, I = index.search(emb, top_k)
    resultados = [mapping[idx] for idx in I[0]]
    return resultados, D[0].tolist()

@app.route("/buscar-producto", methods=["POST"])
def buscar_producto_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    resultados, puntajes = buscar_producto(tmp_path)
    return jsonify({"resultados": resultados})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
