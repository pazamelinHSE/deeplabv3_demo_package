FROM python:3.9 as python
ENV PYTHONUNBUFFERED=true
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# install opencv dependencies
RUN set -x \
    && apt-get update -y \
    && apt-get dist-upgrade -y \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 

# RUN add-apt-repository ppa:mc3man/trusty-media
# RUN apt-get update && apt-get install ffmpeg -y

COPY . .
COPY pyproject.toml /app

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

RUN pip3 install streamlit

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "/app/deeplabv3_demo/streamlit_demo.py", "--server.port=8501"]