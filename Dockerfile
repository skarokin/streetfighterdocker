FROM python:3.8.8

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    build-essential \
    python-opengl \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /app

# Split pip installations with verbose logging
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -v jupyter
RUN pip install --no-cache-dir -v opencv-python-headless

WORKDIR /app
COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]