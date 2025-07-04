/*
 * panel-overview.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-27 18:23:38
 * @modified: 2025-03-29 15:29:50
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import {FormWithLabel} from '@/components/form-with-label';
import {LoadingIndicator} from '@/components/loading-indicator';
import VisOverview from '@/components/vis/vis-overview';
import {datasetConfig} from '@/config/datasets';
import {datasetAtom, paramsAtom, selectionAtom} from '@/store/atom';
import {Dataset} from '@/types';
import {Button, Chip, MenuItem, Select} from '@mui/material';
import {useAtom} from 'jotai';
import * as React from 'react';
import {CheckCircle, Close} from '@mui/icons-material';
import {produce} from 'immer';
import LabelPortion from '@/components/vis/label-portion';
import {useTsInstanceData} from '@/hooks/useTsInstanceData';
import {SignatureList} from '@/components/signature-list';
import ClusterView from '@/components/cluster-view';

export interface IPanelOverviewProps {
}

export function PanelOverview(props: IPanelOverviewProps) {
  const [dataset, setDataset] = useAtom(datasetAtom);
  const [params, setParams] = useAtom(paramsAtom);
  const [selection, setSelection] = useAtom(selectionAtom);
  const {data: tsData, loading: tsLoading, error: tsError} = useTsInstanceData();
  const [followZoom, setFollowZoom] = React.useState(true);

  const dsConfig = datasetConfig.find((d) => d.name === dataset);
  const numPos = tsData?.filter((one) => one.pos).length;
  const numNeg = tsData?.filter((one) => one.neg).length;

  // const [clustersLocal, setClustersLocal] = React.useState<number>(params.numClusters);

  const [thresholdLocal, setThresholdLocal] = React.useState<number>(selection.filterThereshold);

  const handleLabelChange = React.useCallback((label: 'pos' | 'neg') => {
    setParams(produce((draft) => {
      if (label === 'pos') {
        draft.posEnabled = !draft.posEnabled;
      } else {
        draft.negEnabled = !draft.negEnabled;
      }
    }));
    setSelection(produce((draft) => {
      draft.shapelets = undefined;
      draft.selectedInstance = undefined;
    }));
  }, [setParams]);

  const handleCompute = React.useCallback(() => {
    console.log('compute');
    // setParams(produce((draft) => {
    //   draft.numClusters = clustersLocal;
    // }));
    setSelection(produce((draft) => {
      // draft.clusters = undefined;
      draft.shapelets = undefined;
    }));
  }, [setParams]);

  return (
    <div className={'w-full h-full flex flex-col gap-2 px-3 py-2 flex-initial'}>
      <div className={'flex flex-row gap-2 w-full items-center'}>
        <div className={'panel-title'}>Overview</div>
        <div className={'flex-1'} />
        {/* <Button variant={'outlined'} color={'primary'} onClick={handleCompute}>
          Compute
        </Button> */}
      </div>
      <div className={'flex flex-row gap-2 w-full'}>
        <FormWithLabel label={'Dataset:'}
          className={'flex-1'}
        >
          <Select fullWidth value={dataset} onChange={(e) => {
            setDataset(e.target.value as Dataset);
            setParams(produce((draft) => {
              draft.posEnabled = true;
              draft.negEnabled = true;
            }));
            setSelection(produce((draft) => {
              draft.shapelets = undefined;
              draft.selectedInstance = undefined;
            }));
          }}>
            {datasetConfig.map((one) => (
              <MenuItem key={one.name} value={one.name}>{one.displayName}</MenuItem>
            ))}
          </Select>
        </FormWithLabel>
        <FormWithLabel label={'Labels:'} className={'flex-1'}>
          {tsLoading && <LoadingIndicator />}
          {!tsLoading && tsData && <div className={'flex-row gap-2 flex items-center'}>
            <Chip label={`${dsConfig?.posLabel} (${numPos})`}
              variant={params.posEnabled ? 'filled' : 'outlined'} size={'small'}
              color={'info'} deleteIcon={
                params.posEnabled ? <CheckCircle /> : <Close />
              } onDelete={() => handleLabelChange('pos')}
              onClick={() => handleLabelChange('pos')} />
            <Chip label={`${dsConfig?.negLabel} (${numNeg})`}
              variant={params.negEnabled ? 'filled' : 'outlined'} size={'small'}
              color={'warning'} deleteIcon={
                params.negEnabled ? <CheckCircle /> : <Close />
              } onDelete={() => handleLabelChange('neg')}
              onClick={() => handleLabelChange('neg')} />
            <LabelPortion posCount={numPos} negCount={numNeg} />
          </div>}
        </FormWithLabel>
      </div>

      <div className={'flex flex-row gap-2 w-full'}>
        {/* <FormWithLabel label={'# Clusters:'} className={'flex-1'}>
          <Slider value={clustersLocal}
            step={1} min={1} max={10}
            marks
            valueLabelDisplay={'auto'}
            onChange={(e, v) => setClustersLocal(v as number)} />
        </FormWithLabel>
        <FormWithLabel label={'# Shapelets:'} className={'flex-1'}>
          <Slider value={params.numShaplets}
            step={1} min={5} max={20}
            marks
            valueLabelDisplay={'auto'}
            onChange={(e, v) => setParams(produce((draft) => {
              draft.numShaplets = v as number;
            }))} />
        </FormWithLabel> */}

      </div>

      <VisOverview />

      <SignatureList />

      <ClusterView />

    </div>
  );
}
