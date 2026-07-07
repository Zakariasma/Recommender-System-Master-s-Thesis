FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY chap_5/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY chap_5 ./chap_5

EXPOSE 8000

CMD ["uvicorn", "chap_5.api.main:app", "--host", "0.0.0.0", "--port", "8000"]