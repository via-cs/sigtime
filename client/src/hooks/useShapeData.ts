/*
 * useShapeData.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-03-19 15:35:12
 * @modified: 2025-03-25 18:35:40
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import {datasetAtom, paramsAtom} from '@/store/atom';

import {useShapeInfo, useTransformData} from '@/service/api';
import {selectionAtom} from '@/store/atom';
import {useAtomValue} from 'jotai';
import {IProcessedShapeData} from '@/types';
import _ from 'lodash';
import config from '@/config';

export const useShapeData = (): {
  data: IProcessedShapeData[];
  loading: boolean;
  error: Error | null;
} => {
  const dataset = useAtomValue(datasetAtom);
  const params = useAtomValue(paramsAtom);
  const selection = useAtomValue(selectionAtom);
  const threshold = selection.filterThereshold;
  const {data: spData, isLoading: spLoading, error: spError} = useShapeInfo(dataset);
  const {data: tfData, isLoading: tfLoading, error: tfError} = useTransformData(dataset);

  if (!_.every([spData, tfData])) {
    return {
      data: [],
      loading: spLoading || tfLoading,
      error: spError || tfError,
    };
  }

  const res: IProcessedShapeData[] = spData
      .shapes
      .slice(0, config.maxShapelets)
      .map((shape, idx) => {
        return {
          ...shape,
          count: tfData.data.filter((one) => one[idx] <= threshold).length,
        };
      });

  return {
    data: res,
    loading: spLoading || tfLoading,
    error: spError || tfError,
  };
};
