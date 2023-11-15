FROM python:3.7-slim

RUN python -m pip install rasa
RUN python -m pip install pyvi
WORKDIR /app
COPY . .
RUN rasa train --config configs/config-light.yml

USER 1001

ENTRYPOINT [ "rasa" ]

CMD [ "run","--enable-api","--port","8080","--cors","'*'" ]