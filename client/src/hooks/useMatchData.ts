/*
 * useMatchData.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-03-26 15:50:50
 * @modified: 2025-03-26 16:31:34
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import {useMatchingLocation, useShapeInfo} from '@/service/api';
import {datasetAtom, selectionAtom} from '@/store/atom';
import {IProcessedMatchItem} from '@/types/api';
import {useAtom} from 'jotai';

export const useMatchData = () => {
  const [dataset] = useAtom(datasetAtom);
  const [selection] = useAtom(selectionAtom);
  const {data: matchData, isLoading: matchLoading, error: matchError} = useMatchingLocation(dataset, selection.selectedInstance || 1);
  const {data: spData, isLoading: spLoading, error: spError} = useShapeInfo(dataset);


  if (!matchData || !spData) {
    return {
      data: [],
      loading: matchLoading || spLoading,
      error: matchError || spError,
    };
  }

  const processedMatchData: IProcessedMatchItem[] = matchData?.data.map((d, i) => ({
    ...d,
    vals: spData.shapes[i].vals,
  })) || [];

  return {
    data: processedMatchData,
    loading: matchLoading,
    error: matchError,
  };
};
