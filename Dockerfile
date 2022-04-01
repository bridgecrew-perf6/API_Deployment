FROM python:3.8

RUN mkdir  /app

COPY . /app

RUN rm -rf /app/env

RUN pip3 install -r requirements.txt

CMD python3 app.py
