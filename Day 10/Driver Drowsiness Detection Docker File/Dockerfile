FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=200 torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=200 -r requirements.txt

COPY . .

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]