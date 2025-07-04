/*
 * ts-overview.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-04 18:21:33
 * @modified: 2025-03-30 16:18:54
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

// @ts-nocheck
import * as React from 'react';
import * as d3 from 'd3';
import {paramsAtom, selectionAtom, windowWidthAtom, zoomStatusAtom} from '@/store/atom';
import {useAtom, useSetAtom} from 'jotai';
import {useTsInstanceData} from '@/hooks/useTsInstanceData';
import {IconButton, MenuItem, Select, Slider} from '@mui/material';
import {MyLocation} from '@mui/icons-material';
import InstanceRenderer from '@/lib/instance-renderer';
import {debounce, throttle} from 'lodash';

export interface IVisOverviewProps {
}

export default function VisOverview(props: IVisOverviewProps) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [windowWidth] = useAtom(windowWidthAtom);
  const {data: tsData, loading: tsLoading} = useTsInstanceData();
  const [params, setParams] = useAtom(paramsAtom);
  const [selection] = useAtom(selectionAtom);
  const setZoomStatus = useSetAtom(zoomStatusAtom);

  const renderer = React.useRef<InstanceRenderer>(null);

  React.useEffect(() => {
    const svg = d3.select(svgRef.current);
    if (!svg || tsLoading || !tsData) return;
    if (!renderer.current) {
      renderer.current = new InstanceRenderer(svg);
      const throttledSetZoomStatus = debounce((zoomStatus: IZoomStatus) => {
        setZoomStatus(zoomStatus);
      }, 300);
      renderer.current.onZoomChange = throttledSetZoomStatus;
    }

    renderer.current.setRenderConfig({
      allInstances: tsData,
      params,
      selection,
      numTicks: 10,
      enableZoom: true,
      externalZoomStatus: undefined,
      useRaw: true,
    });
    renderer.current.throttledRender();
  }, [tsData, tsLoading, params, selection, setZoomStatus]);

  return (
    <div className={'w-full h-[35vh] border-1 border-gray-300 relative rounded-sm'}>
      <svg ref={svgRef} id={'svg-overview'} className={'absolute top-0 left-0 w-full h-full cursor-crosshair rounded-sm'}></svg>
      <div className={'absolute top-1 right-1 flex flex-row gap-2 items-center'} id={'reset-btn'}>
        <div className={'text-xs text-gray-500 mr-1'}>Curve Type:</div>
        <Select
          sx={{width: '90px'}}
          size={'small'}
          value={params.curveType}
          onChange={(e) => {
            setParams({...params, curveType: e.target.value});
          }}
        >
          <MenuItem value={'linear'}>Linear</MenuItem>
          <MenuItem value={'curve'}>Curve</MenuItem>
          <MenuItem value={'step'}>Step</MenuItem>
        </Select>
        <div className={'text-xs text-gray-500 mr-1'}>Line Opacity:</div>
        <Slider
          sx={{width: '80px'}}
          min={0.01} max={0.8} step={0.01}
          valueLabelDisplay={'auto'}
          value={params.lineOpacity}
          onChange={(event, value) => {
            setParams({...params, lineOpacity: value});
          }}
        />
        {/* Horizontal Zoom */}
        <div className={'text-xs text-gray-500 mr-1'}>Horizontal Zoom:</div>
        <Slider
          sx={{width: '80px'}}
          min={1} max={5} step={0.1}
          valueLabelDisplay={'auto'}
          value={params.horizontalZoom}
          onChange={(event, value) => {
            setParams({...params, horizontalZoom: value});
          }}
        />
        <IconButton size={'small'} onClick={() => {
          setParams({...params, horizontalZoom: 1, lineOpacity: 0.2});
          renderer.current?.resetZoom();
        }}>
          <MyLocation />
        </IconButton>
      </div>
    </div>
  );
}
