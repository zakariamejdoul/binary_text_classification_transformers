# Let's use the image base-notebook to build our image on top of it
FROM python:3.8

LABEL Maintainer="zakaria.mejdoul"

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "./text_classification_with_transformer.py"]