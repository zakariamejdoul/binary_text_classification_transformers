FROM posologytest/docker-train-image:latest

LABEL Maintainer="zakaria.mejdoul"

EXPOSE 4002

CMD ["python", "api.py"]

