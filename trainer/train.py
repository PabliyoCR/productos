import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import time

# ==== Config ====
EXCEL_FILE = "../../DatosArticulos/productos_test.xlsx"
IMAGE_DIR = "../../DatosArticulos/images"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
EPOCHS = 3
LR = 5e-6
OUTPUT_DIR = "./clip_finetuned"
FAILED_IMAGES_FILE = "local_image_fails.txt"

# ==== Dataset ====
class ProductosDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
        self.failed_products_df = pd.DataFrame(columns=df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Incluir 'No_articulo' en el texto
        text = f"{row['No_articulo']} | {row['Nombre']} | {row['Familia']} | {row['Categoria']} | {row['Subcategoria']}"

        image = None
        image_urls = row["ImagenesUrl"].split(',') if pd.notna(row["ImagenesUrl"]) else []
        found_image = False

        for image_url_str in image_urls:
            image_url_str = image_url_str.strip()
            if not image_url_str:
                continue

            # Extraer el nombre del archivo de la URL (si es una URL completa)
            image_filename = os.path.basename(image_url_str)
            image_path = os.path.join(IMAGE_DIR, image_filename)
            
            if not os.path.exists(image_path):
                # Intentar con la ruta original si no se encuentra en IMAGE_DIR
                image_path = image_url_str

            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    found_image = True
                    break
                except Exception as e:
                    print(f"Fallo al cargar la imagen local {image_path}: {e}")
            else:
                print(f"Imagen local no encontrada: {image_path}")

        if not found_image:
            print(f"No se pudo cargar ninguna imagen para el producto: {text}. Este producto será omitido.")
            self.failed_products_df = pd.concat([self.failed_products_df, pd.DataFrame([row])], ignore_index=True)
            with open(FAILED_IMAGES_FILE, "a") as f:
                f.write(f"{text} - No image found\n") # Anotar el nombre del producto o texto como referencia
            return None  # Retornar None para omitir este producto

        # Procesamiento (sin squeeze, con padding fijo)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77  # longitud típica de CLIP
        )

        return inputs

# ==== Entrenamiento ====
def train():
    df = pd.read_excel(EXCEL_FILE)

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    dataset = ProductosDataset(df, processor)

    # Collate fn para unir los tensores del batch correctamente
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]  # Filtrar los None
        if not batch:
            return None
        batch_out = {}
        for key in batch[0]:
            batch_out[key] = torch.cat([d[key] for d in batch], dim=0)
        return batch_out

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            if batch is None:  # Saltar si el batch está vacío después de filtrar None
                continue
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )

            logits_per_image = outputs.logits_per_image
            labels = torch.arange(len(logits_per_image), device=device)

            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    # Guardar productos fallidos al final del entrenamiento
    if not dataset.failed_products_df.empty:
        dataset.failed_products_df.to_excel("failed_products_during_training.xlsx", index=False)
        print(f"Se han guardado los productos fallidos en 'failed_products_during_training.xlsx'")

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ Modelo guardado en {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
