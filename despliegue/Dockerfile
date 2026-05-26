# Dockerfile - Tablero Saber 11 Antioquia (Proyecto 2 ACTD)
#
# Empaqueta la app Dash + modelos pre-entrenados + dataset para correr
# en una instancia EC2 (Ubuntu) expuesta en el puerto 8050.
#
# Build:   sudo docker build -f despliegue/Dockerfile -t saber11:latest .
# Run:     sudo docker run -d -p 8050:8050 --name saber11 saber11:latest

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# Sistema: dependencias mínimas que TensorFlow/sklearn necesitan en runtime
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependencias Python (separadas del código para aprovechar caching de capas)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Código de la aplicación + datos + modelos pre-entrenados
# ---------------------------------------------------------------------------
COPY app.py .
COPY Analysis/ ./Analysis/
COPY pages/    ./pages/
COPY assets/   ./assets/
COPY Data/     ./Data/
COPY models/   ./models/
COPY mlruns/   ./mlruns/

# Linux es case-sensitive: el codigo de Q2 usa "data/" en minuscula mientras
# el resto del repo usa "Data/" en mayuscula. En Windows no se nota porque
# el filesystem es case-insensitive. Symlink para que ambos accesos funcionen.
RUN ln -s Data data

# Variable de entorno para que TensorFlow no llene los logs con warnings
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1

# Puerto del tablero Dash
EXPOSE 8050

# Comando de arranque
CMD ["python", "app.py"]
