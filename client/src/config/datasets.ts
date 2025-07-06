/*
 * datasets.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-27 18:28:12
 * @modified: 2025-02-05 15:43:23
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import {IDatasetConfig} from '@/types';

export const datasetConfig: IDatasetConfig[] = [
  // {
  //   name: 'GunPoint',
  //   displayName: 'Gun Point',
  //   posValue: 1,
  //   negValue: 2,
  //   posLabel: 'Gun',
  //   negLabel: 'Point',
  // },
  {
    name: 'Robot',
    displayName: 'Robot',
    posValue: '1',
    negValue: '0',
    posLabel: 'Hard',
    negLabel: 'Smooth',
  },
  {
    name: 'robot_1',
    displayName: 'Robot FCN',
    posValue: '1',
    negValue: '0',
    posLabel: 'Hard',
    negLabel: 'Smooth',
  },
  // {
  //   name: 'ECG200',
  //   displayName: 'ECG 200',
  //   posValue: '1',
  //   negValue: '0',
  //   posLabel: 'Normal',
  //   negLabel: 'Ischemia',
  // },
  
  {
    name: 'preterm',
    displayName: 'Preterm dataset',
    posValue: '1',
    negValue: '0',
    posLabel: 'Preterm',
    negLabel: 'No Preterm  ',
  },
  // {
  //   name: 'ECG5000_demo',
  //   displayName: 'ECG 5000 demo',
  //   posValue: '1',
  //   negValue: '0',
  //   posLabel: 'Abnormal',
  //   negLabel: 'Normal',
  // },
  {
    name: 'ECG5000_New',
    displayName: 'ECG 5000',
    posValue: '1',
    negValue: '0',
    posLabel: 'Abnormal',
    negLabel: 'Normal',
  },
  {
    name: 'ECG200_Norm',
    displayName: 'ECG 200',
    posValue: '1',
    negValue: '0',
    posLabel: 'Normal',
    negLabel: 'Ischemia',
  },
  // {
  //   name: 'ECG200_JOINT_20',
  //   displayName: 'ECG 200 Joint',
  //   posValue: '1',
  //   negValue: '0',
  //   posLabel: 'Normal',
  //   negLabel: 'Ischemia',
  // },
  {
    name: 'Strawberry',
    displayName: 'Strawberry',
    posValue: '1',
    negValue: '0',
    posLabel: 'Normal',
    negLabel: 'Abnormal',
  }

];
