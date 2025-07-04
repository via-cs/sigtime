/*
 * panel-ts.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-03-19 18:45:00
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import * as React from 'react';
import {useAtom, useAtomValue} from 'jotai';
import {paramsAtom, selectionAtom, windowWidthAtom, datasetAtom} from '@/store/atom';
import {useShapeData} from '@/hooks/useShapeData';
import {useTsInstanceData} from '@/hooks/useTsInstanceData';
import {LoadingIndicator} from '@/components/loading-indicator';
import TsRenderer, {TsRendererMode} from '@/lib/ts-renderer';
import {Button, ButtonGroup, Checkbox, FormControlLabel, TextField} from '@mui/material';
import {cn, toggleArrayItem} from '@/utils';
import * as d3 from 'd3';
import {colors} from '@/config/colors';
import {datasetConfig} from '@/config/datasets';
import {produce} from 'immer';
import {ISortConfig} from '@/types';
import config from '@/config';

export interface IPanelTsProps {}

export default function PanelTs(props: IPanelTsProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const svgRef = React.useRef<SVGSVGElement>(null);
  const renderer = React.useRef<TsRenderer | null>(null);
  const windowWidth = useAtomValue(windowWidthAtom);
  const [mode, setMode] = React.useState<TsRendererMode>('overview');
  const [sort, setSort] = React.useState<ISortConfig>({
    key: undefined,
    order: 'asc',
  });
  const [customFilter, setCustomFilter] = React.useState<string>('');
  const dataset = useAtomValue(datasetAtom);

  const [params, setParams] = useAtom(paramsAtom);
  const [selection, setSelection] = useAtom(selectionAtom);
  const {data: tsData, loading: tsLoading, error: tsError} = useTsInstanceData();
  const {data: shapeData, loading: shapeLoading, error: shapeError} = useShapeData();
  const curDatasetConfig = React.useMemo(() => {
    return datasetConfig.find((d) => d.name === dataset);
  }, [dataset]);

  // Handle window resize
  React.useEffect(() => {
    if (containerRef.current && svgRef.current && renderer.current) {
      renderer.current.destroy();
      renderer.current.render();
    }
  }, [windowWidth]);

  const attachRendererEvents = React.useCallback(() => {
    if (!renderer.current) {
      return;
    }
    renderer.current.onClickShapelet = function(shapeletIdx, mode) {
      console.log('onClickShapelet', shapeletIdx, mode);
      if (mode === 'overview') {
        setSelection(produce((draft) => {
          draft.shapelets = toggleArrayItem(draft.shapelets || [], shapeletIdx);
        }));
      } else {
        handleSort(`shapelet`, shapeletIdx);
      }
    };
  }, [setSelection, mode]);

  // renderer initialization
  React.useEffect(() => {
    if (svgRef.current && !renderer.current) {
      renderer.current = new TsRenderer(svgRef.current, {
        setFilteredInstances: undefined,
      });
      attachRendererEvents();
    }
    return () => {
      if (renderer.current) {
        renderer.current.destroy();
      }
    };
  }, [svgRef, renderer, attachRendererEvents]);

  // renderer refresh
  React.useEffect(() => {
    if (!renderer.current || !tsData || !shapeData || !params || !selection || tsData.length === 0) {
      return;
    }
    renderer.current.setRenderConfig({
      mode: mode,
      allInstances: tsData,
      shapelets: shapeData,
      params: params,
      selection: selection,
      sort: sort,
    });
    renderer.current.render();
  }, [renderer, tsData, shapeData, params, selection, mode, sort]);

  const isLoading = tsLoading || shapeLoading;
  const hasError = tsError || shapeError;

  const scaleTfWidth = React.useMemo(() => {
    return d3.scaleLinear()
        .domain([1.2, 0])
        .range([0, (renderer.current?.DETAILED_SHAPELET_WIDTH || 60) * 0.8]);
  }, [mode]);
  const dimensionWidth = React.useMemo(() => {
    const r = renderer.current;
    if (!r) return 0;
    return (r.width - r.DETAILED_MARGIN_LEFT - r.DETAILED_MARGIN_RIGHT) / (r.dimensions.length - 1);
  }, [renderer.current?.x, mode]);


  const filteredInstances = React.useMemo(() => {
    if (!renderer.current) {
      return [];
    }
    const instances = renderer.current.filterInstances();
    if (!customFilter) return instances;

    if (customFilter.toLowerCase().startsWith('s')) {
      const clusterNum = parseInt(customFilter.slice(1));
      if (!isNaN(clusterNum)) {
        return instances.filter((instance) => instance.closestShapelet === clusterNum - 1);
      }
    }

    const filterNum = parseInt(customFilter);
    if (!isNaN(filterNum)) {
      return instances.filter((instance) => instance.iid === filterNum);
    }

    return instances;
  }, [renderer.current, params, selection, mode, customFilter]);

  const sortedFilteredInstances = React.useMemo(() => {
    const ret = [...filteredInstances];
    if (!sort.key) return ret.slice(0, config.maxDetailedInstances);
    return ret.sort((a, b) => {
      let comparison = 0;
      switch (sort.key) {
        case 'id':
          comparison = a.iid - b.iid;
          break;
        case 'cluster':
          comparison = a.closestShapelet - b.closestShapelet;
          break;
        case 'label':
          comparison = (a.pos === b.pos) ? 0 : (a.pos ? 1 : -1);
          break;
        case 'shapelet':
          comparison = -a.tf[sort.sid || 0] + b.tf[sort.sid || 0];
          break;
      }
      return sort.order === 'asc' ? comparison : -comparison;
    }).slice(0, config.maxDetailedInstances);
  }, [filteredInstances, sort]);

  const handleSort = (key: 'id' | 'cluster' | 'label' | `shapelet`, sid?: number) => {
    setSort((prev) => {
      if (prev.key === key) {
        if (prev.order === 'asc') return {key, order: 'desc', sid};
        if (prev.order === 'desc') return {key: undefined, order: 'asc', sid};
      }
      return {key, order: 'asc', sid};
    });
  };

  return (
    <div className="w-full h-full flex flex-col gap-2 px-3 py-2">
      <div className="panel-title">Time Series Signatures</div>
      <div className="flex flex-row gap-4">
        <ButtonGroup size={'small'}>
          <Button onClick={() => setMode('overview')} variant={mode === 'overview' ? 'contained' : 'outlined'}>Relationship</Button>
          <Button onClick={() => setMode('individual')} variant={mode === 'individual' ? 'contained' : 'outlined'}>Matrix</Button>
        </ButtonGroup>
        {mode === 'overview' && (
          <>
            <FormControlLabel
              control={
                <Checkbox
                  sx={{py: 0, pr: 0.5}}
                  checked={params.showConnections}
                  onChange={(e) => setParams({...params, showConnections: e.target.checked})}
                  name="showConnections"
                  color="primary"
                />
              }
              label="Show Connections"
            />
          </>
        )}
        <TextField
          type="text"
          size="small"
          variant="standard"
          placeholder={'Filter by id or cluster'}
          value={customFilter}
          onChange={(e) => setCustomFilter(e.target.value)}
          sx={{
            display: mode === 'individual' ? 'flex' : 'none',
          }}
        />
        <div className="flex-1" />
        {
          mode == 'individual' && filteredInstances.length > config.maxDetailedInstances && (
            <div className="flex flex-row gap-2 items-center text-red-800">
              Only showing first {config.maxDetailedInstances} instances.
            </div>
          )
        }
        <div className="flex flex-row gap-2 items-center">
          Filtered Instances: {filteredInstances.length}
        </div>
      </div>
      <div className="flex-1 relative" ref={containerRef}>
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-70 z-10">
            <LoadingIndicator />
          </div>
        )}
        {hasError && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-70 z-10">
            <div className="text-red-800">Error loading data</div>
          </div>
        )}

        <svg ref={svgRef} className="w-full h-full border border-gray-300" preserveAspectRatio="none">
        </svg>

        <div className={cn(mode === 'overview' && 'hidden')}>
          <div
            style={{
              top: (renderer.current?.margin?.top || 60) + 10,
              height: `calc(100% - ${(renderer.current?.margin?.top || 60) + (renderer.current?.margin?.bottom || 20) + 10}px)`,
            }}
            className={cn(
                'absolute left-0 right-0 w-full overflow-y-auto overflow-x-clip',
                'flex flex-col gap-1',
            )}>
            {sortedFilteredInstances.map((instance) => (
              <div key={instance.iid} className='flex flex-row w-full items-center instance-item py-1'
                style={{
                  backgroundColor: selection.selectedInstance === instance.iid ? colors.highlightLight2 : 'transparent',
                }}
                onClick={() => {
                  setSelection((prev) => ({...prev, selectedInstance: instance.iid}));
                }}
              >
                <div style={{
                  width: (renderer.current?.DETAILED_MARGIN_LEFT || 240),
                }}
                className='flex flex-row gap-8 items-center font-semibold shrink-0 text-sm'>
                  <div className='w-10 text-right'>{instance.iid}</div>
                  <div className='w-8 text-right'>S{instance.closestShapelet + 1}</div>
                  <div className={'w-12 text-right flex flex-row gap-2 items-center'}
                    style={{
                      color: instance.pos ? colors.pos : colors.neg,
                    }}
                  >
                    <div className='flex flex-row gap-2 items-center text-sm'>
                      <svg width={10} height={10}>
                        <circle cx={5} cy={5} r={5} fill={instance.pos ? colors.pos : colors.neg} />
                      </svg>
                      <div>
                        {instance.pos ? curDatasetConfig?.posLabel : curDatasetConfig?.negLabel}
                      </div>
                    </div>
                  </div>
                </div>
                <div className='relative flex flex-1 flex-row items-center justify-center'
                  style={{
                    transform: `translateX(-${dimensionWidth / 2 - 1}px)`,
                    width: `calc(100% - ${renderer.current?.DETAILED_MARGIN_LEFT || 240}px)`,
                  }}
                >
                  {instance.tf.map((tf, i) => (
                    <div key={i} className='absolute flex flex-row items-center justify-center'
                      style={{
                        width: dimensionWidth,
                        left: (i * dimensionWidth),
                      }}
                    >
                      <div
                        className={'rounded-full'}
                        style={{
                          width: scaleTfWidth(tf),
                          height: 7,
                          backgroundColor: instance.pos ? colors.pos : colors.neg,
                        }} />
                    </div>
                  ))}
                </div>
              </div>
            ))}

          </div>
          <div style={{
            width: (renderer.current?.DETAILED_MARGIN_LEFT || 240),
            height: (renderer.current?.margin?.top || 60) + 10,
          }} className='absolute top-0 left-0' >
            <div className='flex flex-row gap-1 items-center font-bold text-sm h-full px-4'>
              <div
                className='w-12 text-left cursor-pointer'
                onClick={() => handleSort('id')}
              >
                ID {sort.key === 'id' && (sort.order === 'asc' ? '↑' : '↓')}
              </div>
              <div
                className='w-20 text-left cursor-pointer '
                onClick={() => handleSort('cluster')}
              >
                Cluster {sort.key === 'cluster' && (sort.order === 'asc' ? '↑' : '↓')}
              </div>
              <div
                className='w-16 text-left cursor-pointer '
                onClick={() => handleSort('label')}
              >
                Label {sort.key === 'label' && (sort.order === 'asc' ? '↑' : '↓')}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
