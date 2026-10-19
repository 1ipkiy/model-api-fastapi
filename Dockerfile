FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY train_model.py .
COPY model ./model

# Train at build time if artifact missing (keeps image self-contained)
RUN if [ ! -f model/spam_clf.joblib ]; then python train_model.py; fi

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
