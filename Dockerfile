FROM python:3.8

RUN mkdir  /app

COPY . /app

RUN rm -rf /app/env

WORKDIR /app

RUN pip3 install -r requirements.txt

CMD ["python", "app.py"]