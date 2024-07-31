FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
COPY . /app/

RUN pip install -U pip && \
  pip install -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
