/*
 * signature-list.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io) / Yu-Chia Huang (ycihuang@ucdavis.edu)
 * @created: 2025-02-05 00:49:56
 * @modified: 2025-08-05
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import {FormWithLabel} from '@/components/form-with-label';
import {LoadingIndicator} from '@/components/loading-indicator';
import VisSigItem from '@/components/vis/vis-sigitem';
import {useShapeData} from '@/hooks/useShapeData';
import {datasetAtom, selectionAtom} from '@/store/atom';
import {ArrowRight} from '@mui/icons-material';
import {Button, FormControlLabel, Radio, RadioGroup} from '@mui/material';
import {produce} from 'immer';
import {useAtom} from 'jotai';
import React from 'react';

export interface ISignatureListProps {

}

export function SignatureList(props: ISignatureListProps) {
  const [dataset, setDataset] = useAtom(datasetAtom);
  const {data: spData, loading: spLoading, error: spError} = useShapeData();
  const [selection, setSelection] = useAtom(selectionAtom);

  const sortedShapelets = React.useMemo(() => {
    if (!spData) return [];
    return spData.sort((a, b) => {
      if (selection.sortBy === 'id') return a.id - b.id;
      return b.count - a.count;
    });
  }, [spData, selection.sortBy]);
  const threshold = 5; // Exclude shapelets with similarity below this threshold

  return (
    <>
      <div className={'flex flex-row gap-2 w-full items-baseline'}>
        <div className={'panel-title-2'}>Signatures</div>
        <div className={'text-xs text-gray-500 italic'}>Scroll to see all <ArrowRight /> </div>
        <div className={'flex-1'} />
        <FormWithLabel label={'Sort By:'} className={'h-6'}>
          <RadioGroup
            row
            value={selection.sortBy}
            onChange={(e) => {
              setSelection(produce((draft) => {
                draft.sortBy = e.target.value as 'id' | 'numSamples';
              }));
            }}
          >
            <FormControlLabel value="id" label="ID" control={<Radio />} />
            <FormControlLabel value="numSamples" label="# Samples" control={<Radio />} />
          </RadioGroup>
        </FormWithLabel>
        <Button variant={'outlined'} sx={{height: '24px'}}
          disabled={!selection.shapelets}
          onClick={() => {
            setSelection(produce((draft) => {
              draft.shapelets = undefined;
            }));
          }}
        >Cancel Selection</Button>
      </div>

      <div className={'flex flex-row w-full gap-2 h-24 no-shrink rounded-sm overflow-x-auto relative p-0 invisible-scrollbar'}>
        {spLoading && <LoadingIndicator fullScreen={true} />}
        {!spLoading && spData && spData
        .filter((shapelet) => !shapelet.sims?.some((value, index) => value < threshold && index < shapelet.id)) 
        .map((shapelet) => 
        {
          return (
            <VisSigItem
              onClick={() => {
                setSelection(produce((draft) => {
                  if (draft.shapelets) {
                    draft.shapelets = draft.shapelets.includes(shapelet.id) ?
                      draft.shapelets.filter((id) => id !== shapelet.id) :
                      [...draft.shapelets, shapelet.id];
                  } else {
                    draft.shapelets = [shapelet.id];
                  }
                }));
              }}
              shapelet={shapelet} key={`${dataset}-shapelet-${shapelet.id}`} />
          )
        })}
      </div>
    </>
  );
}

