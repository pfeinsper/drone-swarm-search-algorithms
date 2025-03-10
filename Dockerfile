FROM rayproject/ray:2.38.0-py312-gpu

RUN sudo apt update && sudo apt install -y libgdal-dev gdal-bin

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir . && pip uninstall -y app

CMD ["/bin/sh"]