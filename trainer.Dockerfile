FROM python:3.8-buster

LABEL Maintainer="zakaria.mejdoul"

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .


CMD ["python", "text_classification_with_transformer.py"]
