# sigtime

## With Docker

```
docker compose up
```

The source code are mounted via docker volume, so the project is hot-reloaded. Note that you will need to rebuild the docker if you modified dependencies (eg. `pnpm i` or `pixi add`).

## Without Docker

### backend

Make sure `pixi` is installed (https://pixi.sh)

```
pixi i
pixi r dev
```

### client

Enable `pnpm` by running `corepack enable` (for node >= 14)

```
pnpm i
pnpm dev
```
