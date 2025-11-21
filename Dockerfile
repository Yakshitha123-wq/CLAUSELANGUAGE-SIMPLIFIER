FROM python:3.10

WORKDIR /app

COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

COPY . .

RUN mkdir -p instance

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
