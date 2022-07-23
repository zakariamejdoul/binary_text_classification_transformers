FROM posologytest/docker-train-image:latest

LABEL Maintainer="zakaria.mejdoul"

ENV FLASK_APP api.py

COPY . .
 
EXPOSE 4002

CMD ["python", "api.py"]

