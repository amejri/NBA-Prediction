FROM python:3.6

ADD requirements.txt /
RUN pip3 install -r requirements.txt

ADD . /app
WORKDIR /app

EXPOSE 5000
CMD ["python3", "app.py"]




