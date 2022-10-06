ARG BASE_IMAGE=python:3.8-buster
FROM $BASE_IMAGE


# install java
RUN apt-get update && apt-get install -y default-jdk-headless && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME "/usr/lib/jvm/default-java"
# install project requirements
COPY src/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
useradd -d /home/kedro -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro

# copy the whole project except what is in .dockerignore
WORKDIR /home/kedro
COPY . .
RUN chown -R kedro:${KEDRO_GID} /home/kedro
USER kedro
RUN chmod -R a+w /home/kedro

EXPOSE 8888

# download spacy modules
RUN python -m spacy download en_core_web_lg; \
python -m spacy download en_core_web_sm; \
python -m spacy download en


CMD ["kedro", "run"]
