FROM python:3.10-slim

# Installa ffmpeg e le dipendenze di sistema necessarie per audio e git
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia e installa le dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il resto del codice del backend
COPY . .

# Hugging Face e Render usano una porta dinamica (PORT), con fallback a 7860
EXPOSE 7860

# Avvia FastAPI leggendo la porta dinamica dell'ambiente cloud
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
