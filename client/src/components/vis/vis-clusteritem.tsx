/*
 * vis-clusteritem.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-11 17:52:40
 * @modified: 2025-03-30 16:18:44
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
// @ts-nocheck
import * as React from 'react';
import {useTsInstanceData} from '@/hooks/useTsInstanceData';
import {paramsAtom, selectionAtom, windowWidthAtom, zoomStatusAtom} from '@/store/atom';
import {useAtomValue} from 'jotai';
import InstanceRenderer from '@/lib/instance-renderer';
import {colors} from '@/config/colors';
import LabelPortion from '@/components/vis/label-portion';

export interface IVisClusterItemProps {
  cluster: number;
  followZoom: boolean;
}

export default function VisClusterItem(props: IVisClusterItemProps) {
  const {data: tsData, loading: tsLoading} = useTsInstanceData();
  const svgRef = React.useRef<SVGSVGElement>(null);
  const renderer = React.useRef<InstanceRenderer>(null);
  const params = useAtomValue(paramsAtom);
  const selection = useAtomValue(selectionAtom);
  const windowWidth = useAtomValue(windowWidthAtom);
  const zoomStatus = useAtomValue(zoomStatusAtom);

  const posCount = tsData.filter((d) => d.tf[props.cluster] <= selection.filterThereshold && d.pos).length;
  const negCount = tsData.filter((d) => d.tf[props.cluster] <= selection.filterThereshold && !d.pos).length;

  React.useEffect(() => {
    if (!svgRef.current || tsLoading) return;
    if (!renderer.current) {
      renderer.current = new InstanceRenderer(svgRef.current);
    }
    const instances = tsData.filter((d) => d.tf[props.cluster] <= selection.filterThereshold);
    renderer.current.setRenderConfig({
      allInstances: instances,
      params,
      selection,
      numTicks: 4,
      margin: {top: 5, right: 5, bottom: 12, left: 25},
      enableZoom: false,
      externalZoomStatus: props.followZoom ? zoomStatus : undefined,
      useRaw: true,
    });
    renderer.current.throttledRender();
  }, [renderer.current, params.curveType, selection.filterThereshold, zoomStatus]);

  return (
    <div className={`w-full h-full flex flex-col relative`}
    >
      <svg ref={svgRef} className={`flex flex-1 ${posCount + negCount === 0 ? 'opacity-0' : ''}`}></svg>
      <div className={`absolute top-0 left-0 h-full w-full flex items-center justify-center ${posCount + negCount > 0 ? 'hidden' : ''}`}>
        <div className={'text-sm text-gray-500'}>No instances found</div>
      </div>
      <div className={'h-[1.35rem] w-full px-1'}>
        <div className={'flex flex-row w-full gap-2 items-center'}>
          <div className={'text-mono-gray'}>S{props.cluster + 1}</div>
          <div className={'flex-1'} />
          <div className={'text-xs'}>
            <span style={{color: colors.pos}}>{posCount}</span>
            <span> / </span>
            <span style={{color: colors.neg}}>{negCount}</span>
          </div>
          <LabelPortion posCount={posCount} negCount={negCount} size={13} />
        </div>
      </div>
    </div>
  );
}
