# SigTime
In this repo, there are three main directories:
```
- backend: For the backend control of the visualization interface.
- client: For the frontend control of the visualization interface.
- ml-alg: For the model training.
```

## Quick Start
**backend**
Make sure `pixi` is installed (https://pixi.sh)

```
cd backend
pixi i
pixi r dev
```

**client**

Enable `pnpm` by running `corepack enable` (for node >= 14)

```
cd client
pnpm i
pnpm dev
```
**You can also run with Docker.**
Make sure you have Docker running.

```
docker compose up
```

The source code is mounted via Docker volume, so the project is hot-reloaded. Note that you will need to rebuild the Docker if you modified dependencies (eg, `pnpm i` or `pixi add`).


## ml-alg
### Environment
Install the essential packages with 
```
cd ml-alg
pip install -r requirements.txt
```
### Quick start
Training sigtime model with respective parameters, for example,
```
python main.py --epochs 100 --datatype public --dataset ECG200
```
The system supports public datasets from https://timeseriesclassification.com.
