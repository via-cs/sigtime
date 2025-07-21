from fastapi import FastAPI, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import csv
import json
import pandas as pd
import numpy as np

from app.utils import get_data_file
from app.models import TimeSeriesReturnModel
from app.models import ShapeReturnModel, TransformReturnModel, MatchingLocation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

origins = ['http://localhost:5670', 'localhost:5670']
shape_num = 10

# enable cross-origin requests, i.e. requests from a different protocol
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)



def time_series_reader(dataset):
    filename1 = get_data_file(f'{dataset}/X_train.csv')
    xtrain = pd.read_csv(filename1)
    filename2 = get_data_file(f'{dataset}/label.csv')
    if dataset == 'ECG5000_New' or dataset == 'ECG5000_demo':
        filename3 = get_data_file(f'{dataset}/script_all.csv')
        raw_data = pd.read_csv(filename3)
        

    info_data = []
    with open(filename2, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(spamreader, None)
        for row in spamreader:
            info_data.append(row)
    all_group = []

    for rowNum, row in enumerate(xtrain.values):
        rowData = []
        for i, value in enumerate(row):
            
            raw = value
            if dataset == 'ECG5000_New' or dataset == 'ECG5000_demo':
                raw = raw_data.values[rowNum, i]
            obj = {
                't': i,
                'val': +float(value),
                'raw': +float(raw),
            }
            rowData.append(obj)
        group = {
            'iid': rowNum,
            'label': info_data[rowNum][0],
            'ts_data': rowData,
        }
        all_group.append(group)
    return all_group



@app.get('/{dataset}/tsdata', )
async def get_time_series(
    dataset: str = Path(..., title="Dataset", description="The name of the dataset to fetch time series data for.")
) -> TimeSeriesReturnModel:
    """
    This is a test endpoint to get time series data
    **Fast API** can display Markdown `code`.

    """
    all_group = time_series_reader(dataset)
    return TimeSeriesReturnModel(data=all_group)

@app.get('/{dataset}/shape_info')
async def get_shape_info(
    dataset: str = Path(..., title="Dataset", description="The name of the dataset to fetch time series data for.")
)-> ShapeReturnModel:
    """
    Return: a list of shapelet info, each of its contains
    This is a test endpoint to get shape data.
    """
    with open(get_data_file(f'./{dataset}/output_shapelet.json')) as f:
        shape_info = json.load(f)
    shape_list = []
    for i, shape in enumerate(shape_info):
        obj = {
            'id': i,
            'len': shape['len'],
            'gain': shape['gain'],
            'vals': shape['wave'][0],
        }
        shape_list.append(obj)

    return ShapeReturnModel(shapes=shape_list)

@app.get('/{dataset}/transform/')
async def get_transform_data(
    dataset: str = Path(..., title="Dataset", description="The name of the dataset to fetch time series data for.")
)-> TransformReturnModel:
    """
    Get the transformed data. Return a 2D matrix indicating the distance from each instance to each shapelet.
    """
    X_transformed = pd.read_csv(get_data_file(f'{dataset}/shapelet_transform.csv'))
    with open(get_data_file(f'./{dataset}/output_shapelet.json')) as f:
        shape_info = json.load(f)
    transformed_value = np.array(X_transformed)
    for i, shape in enumerate(shape_info):
        wave_length = len(shape['wave'][0])
        print(wave_length)
        transformed_value[:, i] = transformed_value[:, i] / wave_length
    # normalized_value = (transformed_value - np.min(transformed_value)) / (np.max(transformed_value) - np.min(transformed_value))
    normalized_value = (transformed_value - np.min(transformed_value, axis=0)) / (np.max(transformed_value, axis=0) - np.min(transformed_value, axis=0) + 1e-8)
    return TransformReturnModel(
        max=np.max(transformed_value),
        min=np.min(transformed_value),
        data=normalized_value.tolist()
    )

@app.get('/{dataset}/matching/')
async def get_shape_match(
    dataset: str = Path(..., title="Dataset", description="The name of the dataset to fetch time series data for."),
    instance_id: int = Query(default=0, title='Instance ID', description='''
        This instance_id should **always** no less than `0`.
''')
)-> MatchingLocation:
    """
    Return: A list of matching pair (s: starting_index, e: ending index) for specific sample ID. 
    This is a test endpoint to get matching location data
    """
    X_transformed = pd.read_csv(get_data_file(f'{dataset}/shapelet_transform.csv'))
    transformed_value = np.array(X_transformed)
    normalized_value = (transformed_value - np.min(transformed_value)) / (np.max(transformed_value) - np.min(transformed_value))
    pos_start = pd.read_csv(get_data_file(f'{dataset}/match_start.csv'))
    pos_end = pd.read_csv(get_data_file(f'{dataset}/match_end.csv'))
    X_position = []
    for j in range(pos_start.shape[1]):
        X_position.append({
            's': pos_start.values[instance_id, j],
            'e': pos_end.values[instance_id, j], 
            'dist': normalized_value[instance_id, j]
        })
    return MatchingLocation(
        data=X_position)

