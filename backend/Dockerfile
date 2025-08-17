FROM python:3.9-slim-bookworm

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install faiss-cpu torch git+https://github.com/openai/CLIP.git numpy Pillow Flask Flask-Cors

EXPOSE 8000

CMD ["python", "server.py"]
