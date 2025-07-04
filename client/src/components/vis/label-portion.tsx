/*
 * label-portion.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-04 17:58:28
 * @modified: 2025-03-19 14:04:59
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import * as React from 'react';
import * as d3 from 'd3';
import {colors} from '@/config/colors';
import {useAtom} from 'jotai';
import {paramsAtom} from '@/store/atom';

export interface ILabelPortionProps {
  size?: number;
  posCount: number;
  negCount: number;
}

export default function LabelPortion(props: ILabelPortionProps) {
  const [params] = useAtom(paramsAtom);

  const size = props.size ?? 24;
  const svgRef = React.useRef<SVGSVGElement>(null);

  React.useEffect(() => {
    const svg = d3.select(svgRef.current);
    if (!svg) return;
    svg.selectAll('*').remove();

    const innerRadius = size / 2 * 0.6;
    const outerRadius = size / 2;
    const posColor = params.posEnabled ? colors.pos : colors.posLight;
    const negColor = params.negEnabled ? colors.neg : colors.negLight;

    const pos = props.posCount / (props.posCount + props.negCount);

    const g = svg.append('g')
        .attr('transform', `translate(${size/2},${size/2})`);

    const posArc = d3.arc()
        .innerRadius(innerRadius)
        .outerRadius(outerRadius)
        .startAngle(0)
        .endAngle(Math.PI * 2 * pos);

    g.append('path')
        .attr('d', posArc)
        .attr('fill', posColor);

    const negArc = d3.arc()
        .innerRadius(innerRadius)
        .outerRadius(outerRadius)
        .startAngle(Math.PI * 2 * pos)
        .endAngle(Math.PI * 2);

    g.append('path')
        .attr('d', negArc)
        .attr('fill', negColor);
  }, [props.posCount, props.negCount, size, params.posEnabled, params.negEnabled]);

  return (
    <div style={{width: size, height: size}}>
      <svg ref={svgRef} width={size} height={size} />
    </div>
  );
}
