/*
 * index.d.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-27 17:25:28
 * @modified: 2025-03-27 09:03:26
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import {IShapeData, IShapeResponse, ITimeSeriesData} from '@/types/api';

export interface IDatasetConfig {
  name: Dataset;
  displayName: string;
  posValue: number | string;
  negValue: number | string;
  posLabel: string;
  negLabel: string;
}

export type Dataset = 'GunPoint' | 'ECG200' | 'Coffee' | 'BirdChicken' | 'Night' | 'Robot' | 'preterm' | 'ECG5000' | 'ECG200_Norm' | 'ECG200_JOINT_20' | 'ECG5000_New' | 'Strawberry' | 'ECG5000_demo' | 'robot_1';
export type CurveType = 'step' | 'linear' | 'curve';

export type AtomSetter<T> = (value: T | ((prev: T) => T)) => void;

export interface ITsDataInstances extends ITimeSeriesData {
  pos: boolean;
  neg: boolean;
  cluster?: number;
  tf: number[]; // distance to each shapelet.
  closestShapelet: number;
};

export interface IProcessedShapeData extends IShapeData {
  count: number;
}

export interface IParams {
  numClusters: number;
  numShaplets: number;
  posEnabled: boolean;
  negEnabled: boolean;
  lineOpacity: number;
  horizontalZoom: number;
  curveType: CurveType;
  showConnections: boolean;
}

export interface ISelection {
  shapelets: number[] | undefined;
  // clusters: number[] | undefined; <- removed. now it is in sync with the shaplets.
  followZoom: boolean;
  filterThereshold: number;
  filterMode: 'intersect' | 'union';
  selectedInstance: number | undefined;
  sortBy: 'id' | 'numSamples';
}

export interface IZoomStatus {
  transform: d3.ZoomTransform;
  width: number;
  height: number;
}

export interface ISortConfig {
  key?: string;
  sid?: number;
  order: 'asc' | 'desc';
}
