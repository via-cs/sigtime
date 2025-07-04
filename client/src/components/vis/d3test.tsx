/*
 * d3test.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-27 18:48:55
 * @modified: 2025-03-19 14:04:59
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import * as React from 'react';
import * as d3 from 'd3';
import {useAtom} from 'jotai';
import {windowWidthAtom} from '@/store/atom';

export interface ID3TestProps {

}

export function D3Test(props: ID3TestProps) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const windowWidth = useAtom(windowWidthAtom)[0];

  React.useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    svg.append('circle')
        .attr('cx', width / 2)
        .attr('cy', height / 2)
        .attr('r', 50)
        .attr('fill', 'gray');
  }, [windowWidth]);

  return (
    <div className={'w-full h-[30em] border-1 border-gray-300'}>
      <svg ref={svgRef} className={'w-full h-full'}>
      </svg>
    </div>
  );
}
