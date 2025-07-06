/*
 * cluster-view.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-05 13:32:37
 * @modified: 2025-03-28 15:05:46
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import VisClusterItem from '@/components/vis/vis-clusteritem';
import {colors} from '@/config/colors';
import {useShapeData} from '@/hooks/useShapeData';
import {useTransformData} from '@/service/api';
import {datasetAtom, paramsAtom, selectionAtom} from '@/store/atom';
import {Button, ButtonGroup, Checkbox, FormControlLabel, Slider} from '@mui/material';
import {produce} from 'immer';
import {useAtom} from 'jotai';
import _ from 'lodash';
import * as React from 'react';

export interface IClusterViewProps {
}

export default function ClusterView(props: IClusterViewProps) {
  const [dataset] = useAtom(datasetAtom);
  const [params] = useAtom(paramsAtom);
  const [selection, setSelection] = useAtom(selectionAtom);
  // const {data: csData, isLoading: csLoading, error: csError} = useClusterData(dataset, params.numClusters);
  const {data: tfData, isLoading: tfLoading, error: tfError} = useTransformData(dataset);
  const {data: spData, loading: spLoading, error: spError} = useShapeData();


  const numberOfShapelets = spData?.length || 0;
  const [thresholdLocal, setThresholdLocal] = React.useState<number>(0.8);


  const setThresholdAtom = React.useCallback((v: number) => {
    setSelection(produce((draft) => {
      draft.filterThereshold = v;
    }));
  }, [setSelection]);


  const throttledSetThresholdAtom = React.useCallback(_.throttle(setThresholdAtom, 600, {'leading': true, 'trailing': true}), [setThresholdAtom]);

  const handleThresholdChange = React.useCallback((v: number) => {
    setThresholdLocal(v);
    throttledSetThresholdAtom(1 - v);
  }, [setThresholdLocal, throttledSetThresholdAtom]);


  return (
    <>
      <div className={'flex flex-row gap-4 w-full justify-center items-center'}>
        <div className={'panel-title-2'}>Clustered Data</div>
        <div className={'flex-1'} />
        <FormControlLabel
          control={<Checkbox sx={{height: '24px'}} checked={selection.followZoom}
            onChange={(e) => setSelection(produce((draft) => {
              draft.followZoom = e.target.checked;
            }))} />}
          label={'Follow Zoom'}
        />
        <div>Threshold: </div>
        <Slider
          value={thresholdLocal}
          sx={{width: '80px'}}
          min={0.25}
          max={0.99}
          size={'small'}
          step={0.01}
          valueLabelDisplay={'auto'}
          valueLabelFormat={(v) => `${(v * 100).toFixed(2)}%`}
          onChange={(e, v) => {
            handleThresholdChange(v as number);
          }}
        />
        <ButtonGroup variant={'outlined'} size={'small'} sx={{height: '24px'}}>
          <Button variant={selection.filterMode === 'intersect' ? 'contained' : 'outlined'}
            onClick={() => setSelection(produce((draft) => {
              draft.filterMode = 'intersect';
            }))}
          >Intersect</Button>
          <Button variant={selection.filterMode === 'union' ? 'contained' : 'outlined'}
            onClick={() => setSelection(produce((draft) => {
              draft.filterMode = 'union';
            }))}
          >Union</Button>
        </ButtonGroup>
      </div>
      <div className={'w-full flex flex-wrap gap-2 flex-1 overflow-y-auto justify-start'}>

        {numberOfShapelets > 0 && Array.from({length: numberOfShapelets}).map((_, i) => (
          <div key={i} className={`border border-gray-300 rounded-sm w-[24%] h-[48%] cluster-container`}
            style={selection.shapelets?.includes(i) ? {
              border: `2px solid ${colors.highlightLight}`,
            } : {}}
            onClick={() => {
              setSelection(produce((draft) => {
                if (draft.shapelets) {
                  draft.shapelets = draft.shapelets.includes(i) ?
                    draft.shapelets.filter((id) => id !== i) :
                    [...draft.shapelets, i];
                } else {
                  draft.shapelets = [i];
                }
              }));
            }}
          >
            <VisClusterItem followZoom={selection.followZoom} cluster={i} />
          </div>
        ))}
        {/* {csData && !csError && !csLoading && csData.clusters.map((cluster) => (
          <div key={cluster.id} className={`border border-gray-300 rounded-sm w-[24%] h-[48%] cluster-container`}
            style={selection.shapelets?.includes(cluster.id) ? {
              border: `2px solid ${colors.highlightLight}`,
            } : {}}
            onClick={() => {
              setSelection(produce((draft) => {
                if (draft.shapelets) {
                  draft.shapelets = draft.shapelets.includes(cluster.id) ?
                    draft.shapelets.filter((id) => id !== cluster.id) :
                    [...draft.shapelets, cluster.id];
                } else {
                  draft.shapelets = [cluster.id];
                }
              }));
            }}
          >
            <VisClusterItem followZoom={selection.followZoom} cluster={cluster.id} />
          </div>
        ))} */}
      </div>
    </>
  );
}
