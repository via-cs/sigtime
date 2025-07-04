FROM ghcr.io/prefix-dev/pixi:0.40.3

WORKDIR /app
COPY pyproject.toml .
COPY pixi.lock .
RUN pixi install --locked
COPY . .
EXPOSE 8000
CMD ["pixi", "r", "dev"]