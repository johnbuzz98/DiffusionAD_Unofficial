FROM python:3.10.10-slim-bullseye

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends sudo git vim wget curl unzip libgl1-mesa-glx

RUN python -m pip install --upgrade pip && \
    rm -rf ~/.cache/pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN rm -rf requirements.txt

CMD ["bash"]

