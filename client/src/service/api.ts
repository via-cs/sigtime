
/*
 * api.ts
 *
 *  useSWR automatically caches the data. Feel free to use it to get data.
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-31 16:01:52
 * @modified: 2025-03-26 16:12:26
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import config from '@/config';
import {IGetIPResponse, ITimeSeriesDataResponse, IClusterReturnModel, IShapeResponse, ITransformResponse, IMatchingResponse} from '@/types/api';
import useSWR from 'swr';

const jsonFetcher = async (url: string) => {
  const res = await fetch(url);
  return res.json();
};

const root = config.apiRoot;

export const useMyIP = () => {
  return useSWR<IGetIPResponse>('https://api.ipify.org?format=json');
};

export const useTimeSeries = (dataset: string) => {
  return useSWR<ITimeSeriesDataResponse>(`${root}/${dataset}/tsdata`, jsonFetcher);
};

export const useClusterData = (dataset: string, numClusters: number = 5) => {
  return useSWR<IClusterReturnModel>(`${root}/${dataset}/cluster/?num_clusters=${numClusters}`, jsonFetcher);
};

export const useShapeInfo = (dataset: string) => {
  const res = useSWR<IShapeResponse>(`${root}/${dataset}/shape_info`);
  if (res.data) {
    for (let i = 0; i < res.data.shapes.length; i++) {
      res.data.shapes[i].vals = res.data.shapes[i].vals.map((val) => +val);
    }
  }
  return res;
};

export const useTransformData = (dataset: string) => {
  return useSWR<ITransformResponse>(`${root}/${dataset}/transform/`);
};

export const useMatchingLocation = (dataset: string, iid: number) => {
  return useSWR<IMatchingResponse>(`${root}/${dataset}/matching/?instance_id=${iid}`, jsonFetcher);
};
