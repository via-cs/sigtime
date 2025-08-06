/*
 * api.d.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-29 13:29:44
 * @modified: 2025-03-30 14:49:00
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

export interface IGetIPResponse {
  ip: string;
}

export interface ITsData {
  t: number;
  val: number;
  raw: number;
}

export interface ITimeSeriesData {
  iid: number;
  label: string;
  ts_data: ITsData[];
}

export interface ITimeSeriesDataResponse {
  data: ITimeSeriesData[];
}

export interface IClusterData {
  id: number;
  x: number;
  y: number;
}

export interface IClusterReturnModel {
  clusters: IClusterData[];
  belongings: number[];
}

export interface IShapeData {
  id: number;
  // value can be both positive and negative
  vals: number[];
  len: number;
  // information gain
  gain: number;
  rank: number, 
  sims: number[];
}

export interface IShapeResponse {
  shapes: IShapeData[];
}

export interface ITransformResponse {
  /**
   * 2d matrix indicating the distance between each shapelet and each instance.
   * Shape: [num_instances, num_shapelets]
   */
  data: number[][];
}

export interface IMatchingRequest {
  instance_id: number;
}

export interface IMatchItem {
  // start index in the dataset
  s: number;
  // end index in the dataset
  e: number;
  // distance to the shapelet
  dist: number;
}

export interface IProcessedMatchItem extends IMatchItem {
  xy?: [number, number][];
  sid?: number;
  vals: number[];
}

export interface IMatchingResponse {
  data: IMatchItem[];
}
