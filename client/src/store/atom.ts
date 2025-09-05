/*
 * index.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-28 00:21:08
 * @modified: 2025-03-27 09:48:59
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import {Dataset, IParams, ISelection, ITsDataInstances, IZoomStatus} from '@/types';
import {atom} from 'jotai';
import * as d3 from 'd3';
import config from '@/config';
// Current selected dataset
export const datasetAtom = atom<Dataset>('preterm_0808');
export const paramsAtom = atom<IParams>({
  numClusters: 8,
  numShaplets: 10,
  posEnabled: true,
  negEnabled: true,
  lineOpacity: 0.2,
  horizontalZoom: 1,
  showConnections: false,
  filterSimilar: false,
  similarThreshold: 1.5,
  curveType: config.defaultCurveType,
});

export const selectionAtom = atom<ISelection>({
  // when undefined, all shaplets and clusters are selected
  shapelets: undefined,
  // clusters: undefined, <- removed. now it is in sync with the shaplets.
  filterThereshold: 0.2,
  filterMode: 'intersect',
  sortBy: 'id',
  followZoom: true,
  selectedInstance: undefined,
});

// Current width of the window
export const windowWidthAtom = atom<number>(0);

export const tsDataAtom = atom<ITsDataInstances | null>(null);

export const zoomStatusAtom = atom<IZoomStatus>({
  transform: d3.zoomIdentity,
  width: 0,
  height: 0,
});

