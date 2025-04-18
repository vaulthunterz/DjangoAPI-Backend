FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 ExpenseCategorizationAPI.wsgi:application