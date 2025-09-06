# SigTime Algorithm

## Before you start

Install the essential packages with

```
pip install -r requirements.txt
```

## Quick start

Training sigtime model with respective parameters, for example,

```
python main.py --epochs 100 --datatype public --dataset ECG200
```

The important parameters:

```
--dataset                   str                 dataset's name (options see
                                                timeseriesclassification.com )                    Default: ECG200
--test_ratio                float               Test ratio for data splitting.                    Default: 0.2
--epochs                    int                 Number of epochs                                  Default: 200
--d_model                   int                 Dimension of the unit in tranformer               Default: 2
--nhead                     int                 Number of attention heads for transformer         Default: 2
                                                in the model
```

The system supports public datasets from https://timeseriesclassification.com.
