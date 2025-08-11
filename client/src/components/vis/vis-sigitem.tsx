/*
 * vis-sigitem.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-05 00:49:09
 * @modified: 2025-03-19 14:31:42
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

// @ts-nocheck
import {colors} from '@/config/colors';
import {selectionAtom} from '@/store/atom';
import {IShapeData} from '@/types/api';
import * as d3 from 'd3';
import {useAtom} from 'jotai';
import * as React from 'react';

export interface IVisSigItemProps {
  shapelet: IShapeData;
  onClick?: () => void;
}

export default function VisSigItem(props: IVisSigItemProps) {
  const svgRef = React.useRef<SVGSVGElement>(null);
  const [selection, setSelection] = useAtom(selectionAtom);
  const selected = selection.shapelets?.includes(props.shapelet.id);
  React.useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);

    svg.selectAll('*').remove();

    const data = props.shapelet.vals;
    const len = data.length;
    const extent = d3.extent(data);
    const height = svg.node().clientHeight;
    const width = svg.node().clientWidth;
    const margin = {top: 10, right: 10, bottom: 10, left: 10};
    const scaleX = d3.scaleLinear()
        .domain([0, len])
        .range([margin.left, width - margin.right]);
    const scaleY = d3.scaleLinear()
        .domain(extent)
        .range([height - margin.bottom, margin.top]);

    const line = d3.line()
        .x((d, i) => scaleX(i))
        .y((d, i) => scaleY(d))
        .curve(d3.curveMonotoneX);

    

    svg.append('path')
        .datum(data)
        .attr('d', line)
        .attr('stroke', colors.shapes[props.shapelet.id % colors.shapes.length])
        .attr('fill', 'none')
        .attr('stroke-width', 3);
    svg.append('g')
      .append('rect')
      .attr('x', width / 2 - 18)
      .attr('y', margin.top - 8)
      .attr('width', 36)
      .attr('height', 18)
      .attr('rx', 4)
      .attr('fill', '#fff')
      .attr('stroke', '#bbb')
      .attr('stroke-width', 1);

    svg.append('g')
      .append('text')
      .attr('x', width / 2)
      .attr('y', margin.top + 5)
      .attr('text-anchor', 'middle')
      .attr('alignment-baseline', 'middle')
      .attr('font-size', 12)
      .attr('fill', '#333')
      .text(len);
  }, [svgRef]);

  return (
    <div className={`w-[9rem] flex flex-col items-center shrink-0 h-full rounded-md border-gray-300 border
       cursor-pointer hover:bg-gray-100 transition-all duration-200`}
    style={selected ? {
      borderColor: colors.highlightLight,
      borderWidth: '2px',
    } : {}}
    onClick={props.onClick}
    >
      <svg className={'flex w-full flex-1'} ref={svgRef}> </svg>
      <div className={'w-full h-[1.5rem] flex items-center justify-center'}>
        <div className={`text-mono-gray`}>
          <span
          // style={{color: colors.shapes[props.shapelet.id]}}
          >
            S{props.shapelet.id + 1}
          </span>
        </div>
      </div>
    </div>
  );
}
