FROM python:3.10-slim
WORKDIR /chatbot-api-example
COPY . /chatbot-api-example


RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
