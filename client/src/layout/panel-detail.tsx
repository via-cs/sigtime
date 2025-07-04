/*
 * panel-detail.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-12 00:27:23
 * @modified: 2025-03-30 16:19:04
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import * as React from 'react';
import {useAtom, useAtomValue} from 'jotai';
import {paramsAtom, selectionAtom, zoomStatusAtom} from '@/store/atom';
import {useTsInstanceData} from '@/hooks/useTsInstanceData';
import InstanceRenderer from '@/lib/instance-renderer';
import {LoadingIndicator} from '@/components/loading-indicator';
import {colors} from '@/config/colors';
import {cn} from '@/utils';
import {useMatchData} from '@/hooks/useMatchData';

export interface IPanelDetailProps {
}

export default function PanelDetail(props: IPanelDetailProps) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const renderer = React.useRef<InstanceRenderer | null>(null);
  const [zoomStatus] = useAtom(zoomStatusAtom);
  const [selection] = useAtom(selectionAtom);
  const params = useAtomValue(paramsAtom);
  const {data: tsData, loading: tsLoading, error: tsError} = useTsInstanceData();
  const {data: matchData, loading: matchLoading, error: matchError} = useMatchData();

  React.useEffect(() => {
    if (svgRef.current && !renderer.current) {
      renderer.current = new InstanceRenderer(svgRef.current);
    }
  }, [svgRef]);

  const selectedInstance = React.useMemo(() => {
    return tsData.find((d) => d.iid === selection.selectedInstance) || null;
  }, [tsData, selection.selectedInstance]);

  React.useEffect(() => {
    if (!renderer.current || !tsData || tsData.length === 0 || !selection.selectedInstance) {
      return;
    }
    // Configure renderer with only the selected instance
    renderer.current.setRenderConfig({
      allInstances: [selectedInstance],
      params,
      selection,
      numTicks: 6,
      margin: {top: 10, right: 0, bottom: 20, left: 20},
      enableZoom: true,
      isIndividual: true,
      matchData: matchData,
      externalZoomStatus: zoomStatus,
      useRaw: false,
    });

    renderer.current.render();
  }, [renderer, tsData, params, selection, zoomStatus, selectedInstance, matchData]);

  return (
    <div className={'w-full h-full flex flex-col gap-2 px-3 py-2 flex-initial'}>
      <div className={'flex flex-row gap-2 items-center'}>
        <div className={'panel-title'}>Sample Detail</div>
        <div className={'flex-1'} />
        {selectedInstance && (
          <>
            <div className={'text-mono-gray'}>Instance ID: {selectedInstance?.iid}</div>
            <svg width={10} height={10}>
              <circle cx={5} cy={5} r={5} fill={selectedInstance?.pos ? colors.pos : colors.neg} />
            </svg>
          </>
        )}
      </div>
      <div className={'flex flex-1 relative w-full h-full'}>
        {tsLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-70 z-10">
            <LoadingIndicator />
          </div>
        )}
        {tsError && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-70 z-10">
            <div className="text-red-800">Error loading data</div>
          </div>
        )}
        <div className={cn(
            'absolute top-0 left-0 flex items-center justify-center w-full h-full',
          selectedInstance ? 'hidden' : 'flex',
        )}>
          <div className={'text-gray-500'}>No instance selected</div>
        </div>
        <svg ref={svgRef} className={cn(
            'w-full h-full',
            selectedInstance ? 'opacity-100' : 'opacity-0',
        )} preserveAspectRatio="none"></svg>

      </div>
    </div>
  );
}
