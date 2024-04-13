FROM python:3.11.5

WORKDIR /app

COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

COPY . /app/
EXPOSE 5000
CMD ["python", "app.py"]

