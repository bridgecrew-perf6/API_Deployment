FROM python:3.8

RUN mkdir  /app

COPY . /app

RUN rm -rf /app/env

CMD python app.py
