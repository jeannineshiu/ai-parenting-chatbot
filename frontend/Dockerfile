FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install gradio>=5.0.0 huggingface_hub requests

CMD ["python", "app.py"]