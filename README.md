# SigTime
In this repo, there are three main directories:
```
- backend
- client
- ml-alg
```
The first two are the website interface and the last one is for training purposes. 

## ml-alg
### Environment
Install the essential packages with 
```
pip install -r requirements.txt
```
### Quick start
Training sigtime model with respective parameters, for example,
```
python main.py --epochs 100 --datatype public --dataset ECG200
```
The system supports public datasets from https://timeseriesclassification.com.

For preterm birth dataset, the command should be:
```
python main.py --epochs 100 --datatype private --dataset preterm
```

## Website Interface
### With Docker

```
docker compose up
```

The source code are mounted via docker volume, so the project is hot-reloaded. Note that you will need to rebuild the docker if you modified dependencies (eg. `pnpm i` or `pixi add`).

### Without Docker
**backend**

Make sure `pixi` is installed (https://pixi.sh)

```
pixi i
pixi r dev
```

**client**

Enable `pnpm` by running `corepack enable` (for node >= 14)

```
pnpm i
pnpm dev
```
