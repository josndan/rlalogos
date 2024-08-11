FROM python:latest

RUN python -m pip install --upgrade pip && \
    pip install clearml-agent

COPY clearml.conf /root/clearml.conf

CMD ["clearml-agent", "daemon", "--queue", "91cb9739e0044f8ab07a566d681cc4fc"]
