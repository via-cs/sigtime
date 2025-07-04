import {useAtom, useAtomValue} from 'jotai';
import {useTimeSeries, useTransformData} from '@/service/api';
import {datasetAtom, paramsAtom, selectionAtom} from '@/store/atom';
// Assuming this is the correct import
import {ITsDataInstances} from '@/types';
import {getDsConfig} from '@/utils/dataset';
import _ from 'lodash';
import {likelyEq} from '@/utils';
import config from '@/config';

export const useTsInstanceData = (): {
  data: Array<ITsDataInstances>;
  loading: boolean;
  error: Error | null;
} => {
  const [dataset] = useAtom(datasetAtom);
  const [params] = useAtom(paramsAtom);
  const selection = useAtomValue(selectionAtom);

  const {data: tsData, isLoading: tsLoading, error: tsError} = useTimeSeries(dataset);
  // const {data: clusterData, isLoading: clusterLoading, error: clusterError} = useClusterData(dataset, params.numClusters);
  const {data: tfData, isLoading: tfLoading, error: tfError} = useTransformData(dataset);

  if (!_.every([tsData, tfData])) {
    return {
      data: [],
      loading: tsLoading || tfLoading,
      error: tsError || tfError,
    };
  }

  const dsConfig = getDsConfig(dataset);
  const res: Array<ITsDataInstances> = tsData.data.map((d) => {
    return {
      ...d,
      pos: likelyEq(d.label, dsConfig.posValue),
      neg: likelyEq(d.label, dsConfig.negValue),
      tf: tfData.data[d.iid].slice(0, config.maxShapelets),
      closestShapelet: 0,
      // cluster: clusterData.belongings[d.iid],
    };
  });

  res.forEach((d) => {
    d.closestShapelet = d.tf.indexOf(Math.min(...d.tf));
  });

  return {
    data: res,
    loading: tsLoading || tfLoading,
    error: tsError || tfError,
  };
};

