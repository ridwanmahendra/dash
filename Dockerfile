FROM python:3.10

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["gunicorn", "-b", "0.0.0.0:6000", "app:app"]