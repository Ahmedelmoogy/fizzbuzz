FROM python:3.8

WORKDIR /app

RUN pip install gunicorn transformers torch  flask 

COPY . /app

WORKDIR /app

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"] 

