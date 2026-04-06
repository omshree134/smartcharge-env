FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"inference\" ]; then python inference.py; else uvicorn service.api:app --host 0.0.0.0 --port ${PORT:-7860}; fi"]
