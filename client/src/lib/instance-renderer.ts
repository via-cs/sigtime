/*
 * instance-renderer.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-11 18:07:21
 * @modified: 2025-03-30 16:18:03
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import {colors} from '@/config/colors';
import {IParams, ISelection, ITsDataInstances, IZoomStatus} from '@/types';
import {IProcessedMatchItem, ITsData} from '@/types/api';
import {chromaToColorRGBA, objEquals} from '@/utils';
import {relativeTransform} from '@/utils/vis';
import * as d3 from 'd3';
import chroma from 'chroma-js';
import _, {debounce, flatten} from 'lodash';

class InstanceRenderer {
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>;
  width: number;
  height: number;
  /// all instances before filtering.
  allInstances: ITsDataInstances[] = [];
  hasConfig: boolean = false;
  enableZoom: boolean = true;
  margin = {top: 20, right: 20, bottom: 20, left: 30};
  zoom: d3.ZoomBehavior<Element, unknown>;
  /// all instances after filtering. these instances are used for rendering.
  instances: ITsDataInstances[];
  canvasWidth: number;
  canvasHeight: number;
  canvasNode: HTMLCanvasElement;
  params: IParams;
  selection: ISelection;
  lastRenderedTransform = d3.zoomIdentity;
  lastRenderedInstanceCount: number = 0;
  canvasScaleX: d3.ScaleLinear<number, number, never>;
  canvasScaleY: d3.ScaleLinear<number, number, never>;
  foreignObj: d3.Selection<d3.BaseType | SVGForeignObjectElement, any, SVGSVGElement, unknown>;
  scaleX: d3.ScaleLinear<number, number, never>;
  scaleY: d3.ScaleLinear<number, number, never>;
  zoomedScaleX: d3.ScaleLinear<number, number, never>;
  zoomedScaleY: d3.ScaleLinear<number, number, never>;
  numTicks: number;
  // If externalZoomStatus is provided, the renderer will use it to zoom.
  externalZoomStatus?: IZoomStatus;
  isIndividual: boolean;
  matchData?: IProcessedMatchItem[];
  useRaw: boolean;

  readonly MAX_ZOOM_LEVEL = 12;
  readonly MIN_ZOOM_LEVEL = 0.5;
  readonly CANVAS_DEBOUNCE = 800;

  dirty: boolean = true;

  onZoomReset?: () => void;
  onZoomChange?: (zoomStatus: IZoomStatus) => void;
  mainG?: d3.Selection<d3.BaseType | SVGGElement, any, SVGSVGElement, unknown>;

  constructor(svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown> | SVGSVGElement) {
    this.svg = svg instanceof SVGSVGElement ? d3.select(svg) : svg;
    this.width = this.svg.node().clientWidth;
    this.height = this.svg.node().clientHeight;
  }

  public destroy() {
    this.svg.selectAll('*').remove();
    this.svg.on('mousemove', null);
    this.svg.on('mouseleave', null);
    if (this.zoom) {
      this.svg.call(this.zoom.transform, d3.zoomIdentity);
    }
  }

  public resetZoom() {
    this.svg.call(this.zoom.transform, d3.zoomIdentity);
    this.lastRenderedTransform = d3.zoomIdentity;
    this.onZoomReset?.();
  }

  private drawCanvasLines2D = (transform?: d3.ZoomTransform) => {
    const xScale = transform.rescaleX(this.canvasScaleX);
    const yScale = transform.rescaleY(this.canvasScaleY);
    const ctx = this.canvasNode.getContext('2d');
    ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);
    const alpha = Math.max(this.params.lineOpacity, 2 / this.instances.length);

    const lineGenerator = d3.line<ITsData>()
        .x((d) => xScale(d.t))
        .y((d) => yScale(this.useRaw ? d.raw : d.val))
        .curve(this.params.curveType === 'step' ? d3.curveStepAfter :
        this.params.curveType === 'linear' ? d3.curveLinear : d3.curveMonotoneX);

    const leftBoundaryT = xScale.invert(0);
    const rightBoundaryT = xScale.invert(this.width - this.margin.right);

    ctx.lineWidth = 2;
    ctx.globalAlpha = alpha;

    this.instances.forEach((instance) => {
      const clippedTsData = instance.ts_data.filter((d) => d.t >= leftBoundaryT && d.t <= rightBoundaryT);
      const pathStr = lineGenerator(clippedTsData);
      if (pathStr) {
        const path = new Path2D(pathStr);
        ctx.strokeStyle = instance.pos ? colors.pos : colors.neg;
        ctx.stroke(path);
      }
      if (transform.k > 7) {
        instance.ts_data.forEach((d) => {
          ctx.beginPath();
          ctx.arc(xScale(d.t), yScale(this.useRaw ? d.raw : d.val), 5, 0, 2 * Math.PI);
          ctx.fillStyle = instance.pos ? colors.pos : colors.neg;
          ctx.fill();
        });
      }
    });
  };

  private drawCanvasLinesWebGL = (transform?: d3.ZoomTransform) => {
    // TODO: implement webgl rendering
    console.error('WebGL rendering is not implemented');
    const devicePixelRatio = window.devicePixelRatio || 1;
    this.canvasNode.width = this.canvasNode.clientWidth * devicePixelRatio;
    this.canvasNode.height = this.canvasNode.clientHeight * devicePixelRatio;

    const posColor = chromaToColorRGBA(chroma(colors.pos));
    const negColor = chromaToColorRGBA(chroma(colors.neg));
  };

  private drawCanvasLines = (externalTransform?: d3.ZoomTransform) => {
    const startTime = performance.now();
    this.foreignObj.attr('transform', null);
    this.mainG?.attr('transform', null);
    const transform = externalTransform || d3.zoomTransform(this.svg.node());

    this.drawCanvasLines2D(transform);

    this.lastRenderedTransform = transform;
    this.lastRenderedInstanceCount = this.instances.length;
    const endTime = performance.now();
    console.log(`full instance rerender time: ${endTime - startTime} ms (instances: ${this.instances.length})`);
  };


  private rescaleAxis = (transform: d3.ZoomTransform) => {
    this.zoomedScaleX = transform.rescaleX(this.scaleX);
    this.zoomedScaleY = transform.rescaleY(this.scaleY);
  };

  private attachZoom = () => {
    const debouncedRedrawCanvas = debounce(() => {
      this.drawCanvasLines();
      if (this.isIndividual) {
        this.drawShapelets();
      }
    }, this.CANVAS_DEBOUNCE / 5);


    this.zoom = d3.zoom()
        .scaleExtent([this.MIN_ZOOM_LEVEL, this.MAX_ZOOM_LEVEL]) // Set zoom limits
        .on('zoom', (event) => {
          const transform = event.transform;
          this.rescaleAxis(transform);
          this.svg.selectAll('.x-axis,.x-textonly')
          // @ts-ignore
              .call(d3.axisBottom(this.zoomedScaleX)
                  .ticks(this.numTicks)
                  .tickSize(-this.height + this.margin.top + this.margin.bottom));
          this.svg.selectAll('.y-axis,.y-textonly')
          // @ts-ignore
              .call(d3.axisLeft(this.zoomedScaleY)
                  .ticks(this.numTicks)
                  .tickSize(-this.width + this.margin.left + this.margin.right));

          const newTransform = relativeTransform(transform, this.lastRenderedTransform);
          this.foreignObj.attr('transform', newTransform.toString());
          this.mainG?.attr('transform', newTransform.toString());
          debouncedRedrawCanvas();
          this.onZoomChange?.({
            transform: event.transform,
            width: this.width,
            height: this.height,
          });
        })
        .on('end', (event) => {
          debouncedRedrawCanvas.cancel();
          this.drawCanvasLines();
          if (this.isIndividual) {
            this.drawShapelets();
          }
          this.onZoomChange?.({
            transform: event.transform,
            width: this.width,
            height: this.height,
          });
        });

    this.svg.call(this.zoom);
    this.onZoomChange?.({
      transform: this.lastRenderedTransform || d3.zoomIdentity,
      width: this.width,
      height: this.height,
    });
  };

  private applyExternalZoom = () => {
    if (this.externalZoomStatus === undefined) return;
    const {transform, width: extWidth, height: extHeight} = this.externalZoomStatus;
    const newx = transform.x / extWidth * this.width;
    const newy = transform.y / extHeight * this.height;
    const newk = transform.k;
    const newTransform = d3.zoomIdentity.translate(newx, newy).scale(newk);

    this.rescaleAxis(newTransform);
    this.drawCanvasLines(newTransform);
    if (this.isIndividual) {
      this.drawShapelets(newTransform);
    }
    this.svg.selectAll('.x-axis,.x-textonly')
        // @ts-ignore
        .call(d3.axisBottom(this.zoomedScaleX)
            .ticks(this.numTicks)
            .tickSize(-this.height + this.margin.top + this.margin.bottom),
        );
    this.svg.selectAll('.y-axis,.y-textonly')
        // @ts-ignore
        .call(d3.axisLeft(this.zoomedScaleY)
            .ticks(this.numTicks)
            .tickSize(-this.width + this.margin.left + this.margin.right),
        );
  };

  public setRenderConfig(config: {
    allInstances: ITsDataInstances[];
    params: IParams;
    selection: ISelection;
    numTicks?: number;
    margin?: {top: number, right: number, bottom: number, left: number};
    enableZoom?: boolean;
    externalZoomStatus?: IZoomStatus;
    isIndividual?: boolean;
    matchData?: IProcessedMatchItem[];
    useRaw?: boolean;
  }) {
    this.dirty = this.params?.curveType !== config.params?.curveType ||
      this.allInstances.length !== config.allInstances.length ||
      this.params?.lineOpacity !== config.params?.lineOpacity ||
      this.params?.posEnabled !== config.params?.posEnabled ||
      this.params?.negEnabled !== config.params?.negEnabled ||
      this.params?.horizontalZoom !== config.params?.horizontalZoom ||
      this.selection?.filterThereshold !== config.selection?.filterThereshold ||
      this.selection?.followZoom !== config.selection?.followZoom ||
      this.useRaw !== config.useRaw ||
      !objEquals(this.externalZoomStatus, config.externalZoomStatus);

    if (this.isIndividual) {
      this.dirty = this.dirty || this.selection.selectedInstance !== config.selection.selectedInstance ||
        !objEquals(this.matchData, config.matchData);
    }

    this.params = config.params;
    this.selection = config.selection;
    this.numTicks = config.numTicks || 10;
    this.margin = config.margin || this.margin;
    this.hasConfig = true;
    this.allInstances = config.allInstances;
    this.enableZoom = config.enableZoom === undefined ? true : config.enableZoom;
    this.externalZoomStatus = config.externalZoomStatus;
    this.isIndividual = config.isIndividual || false;
    this.useRaw = config.useRaw || false;
    this.matchData = config.matchData || [];
  }

  public fullRender() {
    this.destroy();
    this.render();
  }


  private drawCross() {
    const margin = this.margin;
    const crossG = this.svg.selectAll('g.cross').data([null])
        .join('g')
        .attr('class', 'cross')
        .attr('transform', `translate(${margin.left}, ${margin.top})`)
        .attr('opacity', 0);

    const lines = [
      {id: 'cross-line-x', y1: 0, y2: this.height},
      {id: 'cross-line-y', x1: 0, x2: this.width},
    ];

    const texts = [
      {id: 'cross-text-x', y: this.height - margin.bottom - margin.top - 5},
      {id: 'cross-text-y', x: 5},
    ];

    crossG.selectAll('text')
        .data(texts)
        .join('text')
        .attr('id', (d) => d.id)
        .attr('class', 'cross-text')
        .attr('text-anchor', 'start')
        .attr('y', (d) => d.y)
        .attr('x', (d) => d.x)
        .attr('fill', colors.highlight)
        .style('paint-order', 'stroke')
        .style('font-weight', 'bold')
        .style('stroke', 'white')
        .style('stroke-width', 2);

    crossG.selectAll('line')
        .data(lines)
        .join('line')
        .attr('id', (d) => d.id)
        .attr('y1', (d) => d.y1)
        .attr('y2', (d) => d.y2)
        .attr('x1', (d) => d.x1)
        .attr('x2', (d) => d.x2)
        .attr('stroke', colors.highlightLight)
        .attr('stroke-width', 1.5);


    this.svg.on('mousemove', (event) => {
      const [x, y] = d3.pointer(event);
      const t = parseInt(this.zoomedScaleX.invert(x).toFixed(2));
      const val = this.zoomedScaleY.invert(y).toFixed(2);
      crossG.attr('opacity', 1);

      crossG.select('#cross-line-x')
          .attr('x1', x - margin.left)
          .attr('x2', x - margin.left);
      crossG.select('#cross-line-y')
          .attr('y1', y - margin.top)
          .attr('y2', y - margin.top);

      crossG.select('#cross-text-x')
          .text(t)
          .attr('x', x - margin.left + 5);
      crossG.select('#cross-text-y')
          .text(val)
          .attr('y', y - margin.top - 5);
    });

    this.svg.on('mouseleave', () => {
      crossG.transition().duration(200).attr('opacity', 0);
    });
  }

  private drawBlockerAndTextAxis() {
    const margin = this.margin;
    const rects = [
      {x: 0, y: 0, width: this.width, height: margin.top},
      {x: 0, y: this.height - margin.bottom, width: this.width, height: margin.bottom},
      {x: 0, y: margin.top, width: margin.left, height: this.height - margin.top - margin.bottom},
      {x: this.width - margin.right, y: margin.top, width: margin.right, height: this.height - margin.top - margin.bottom},
    ];

    this.svg.selectAll('rect.overflow-blocker')
        .data(rects)
        .join('rect')
        .attr('class', 'overflow-blocker')
        .attr('x', (d) => d.x)
        .attr('y', (d) => d.y)
        .attr('width', (d) => d.width)
        .attr('height', (d) => d.height)
        .attr('fill', 'white');

    // add axis( text only)
    this.svg.selectAll('g.x-textonly')
        .data([Math.random()])
        .join('g')
        .attr('class', 'x-textonly gray-axis')
        .attr('transform', `translate(${0}, ${this.height - margin.bottom})`)
    // @ts-ignore
        .call(d3.axisBottom(this.zoomedScaleX)
            .ticks(this.numTicks)
            .tickSize(0));

    this.svg.selectAll('g.y-textonly')
        .data([Math.random()])
        .join('g')
        .attr('class', 'y-textonly gray-axis')
        .attr('transform', `translate(${margin.left}, ${0})`)
    // @ts-ignore
        .call(d3.axisLeft(this.zoomedScaleY)
            .ticks(this.numTicks)
            .tickSize(0));
  }

  private filterInstances() {
    this.instances = this.allInstances;
    this.instances = this.allInstances.filter((d) => d.pos && this.params.posEnabled || d.neg && this.params.negEnabled);
    this.instances.forEach((d) => {
      d.ts_data.sort((a, b) => a.t - b.t);
    });
  }


  private drawShapelets(externalTransform?: d3.ZoomTransform) {
    if (!this.matchData || this.matchData.length === 0) {
      console.log('Match data not set');
      return;
    }

    const transform = externalTransform || d3.zoomTransform(this.svg.node());
    const xScale = transform.rescaleX(this.canvasScaleX);
    const yScale = transform.rescaleY(this.canvasScaleY);

    const ctx = this.canvasNode.getContext('2d');

    this.matchData.forEach((d, i) => {
      if (d.dist >= this.selection.filterThereshold) {
        return;
      }
      const ys = d.vals.map((v) => yScale(v));
      const xs = Array.from({length: ys.length}, (_, i) => xScale(d.s + (d.e - d.s) / ys.length * i));
      const xy = _.zip(xs, ys) as [number, number][];
      const path = d3.line<[number, number]>()
          .x((d) => d[0])
          .y((d) => d[1]);

      this.matchData[i].xy = _.cloneDeep(xy);
      this.matchData[i].sid = i;
      const path2d = new Path2D(path(xy));
      const color = chroma(colors.shapes[i]).alpha(d.dist / this.selection.filterThereshold + 0.5).rgba().join(',');
      ctx.lineWidth = 3;
      ctx.strokeStyle = `rgba(${color})`;
      ctx.stroke(path2d);
    });

    this.mainG.selectAll('g.annotation').remove();

    const annotationG = this.mainG.selectAll('g.annotation')
        .data(this.matchData?.filter((one) => one.dist <= this.selection.filterThereshold) || [])
        .join('g')
        .attr('class', 'annotation')
        .attr('transform', (d) => {
          return `translate(${(d.xy?.at(0)?.at(0) || 0) - 15}, ${(d.xy?.at(0)?.at(1) || 0) - 15})`;
        });

    // add shapelet name (in circled with bg color as colors.shape[i]) and its distance (in black text
    annotationG.append('circle')
        .attr('r', 10)
        .attr('fill-opacity', 0.5)
        .attr('fill', (d) => colors.shapes[d.sid || 0])
        .attr('stroke', (d) => colors.shapes[d.sid || 0])
        .attr('stroke-width', 2);

    annotationG.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '11px')
        .attr('fill', 'black')
        .attr('font-weight', 'bold')
        .attr('dy', '0.05em')
        .text((d) => `S${d.sid + 1 || 0}`);

    annotationG.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'hanging')
        .attr('font-size', '9px')
        .attr('fill', 'black')
        .attr('y', 15)
        .text((d) => d.dist !== undefined ? `${((1 - d.dist) * 100).toFixed(2)}%` : '');
  }

  public render() {
    if (!this.hasConfig) {
      console.warn('Rendering config not set');
      return;
    }

    if (!this.dirty) {
      console.log('No changes, skipping render');
      return;
    }

    this.filterInstances();

    const margin = this.margin;
    const vals = this.allInstances.map((d) => d.ts_data);
    const dataRange = d3.extent(flatten(vals).map((v) => this.useRaw ? v.raw : v.val));
    const tRange = d3.extent(flatten(vals).map((v) => v.t));
    this.canvasWidth = this.width - margin.left - margin.right;
    this.canvasHeight = this.height - margin.top - margin.bottom;
    const hz = this.params.horizontalZoom;
    const midX = this.canvasWidth / 2;
    const widthX = this.canvasWidth * hz;
    this.scaleX = d3.scaleLinear()
        .domain(tRange)
        .range([margin.left + midX - widthX / 2, margin.left + midX + widthX / 2]);

    this.scaleY = d3.scaleLinear()
        .domain(dataRange)
        .range([this.height - margin.bottom, margin.top]);

    this.canvasScaleX = d3.scaleLinear()
        .domain(tRange)
        .range([midX - widthX / 2, midX + widthX / 2]);

    this.canvasScaleY = d3.scaleLinear()
        .domain(dataRange)
        .range([this.canvasHeight, 0]);

    this.zoomedScaleX = this.scaleX;
    this.zoomedScaleY = this.scaleY;

    this.svg.selectAll('g.x-axis')
        .data([null])
        .join('g')
        .attr('class', 'x-axis gray-axis')
        .attr('transform', `translate(${0}, ${this.height - margin.bottom})`)
    // @ts-ignore
        .call(d3.axisBottom(this.scaleX)
            .ticks(this.numTicks)
            .tickSize(-this.height + margin.top + margin.bottom));

    this.svg.selectAll('g.y-axis')
        .data([null])
        .join('g')
        .attr('class', 'y-axis gray-axis')
        .attr('transform', `translate(${margin.left}, ${0})`)
    // @ts-ignore
        .call(d3.axisLeft(this.scaleY)
            .ticks(this.numTicks)
            .tickSize(-this.width + margin.left + margin.right));

    this.foreignObj = this.svg.selectAll('foreignObject.canvas-container').data([1])
        .join('foreignObject')
        .attr('class', 'canvas-container')
        .attr('x', margin.left)
        .attr('y', margin.top)
        .attr('width', this.canvasWidth)
        .attr('height', this.canvasHeight);

    this.canvasNode = this.foreignObj.selectAll('.main-canvas').data([1])
        .join('xhtml:canvas')
        .attr('class', 'main-canvas')
        .attr('width', this.canvasWidth)
        .attr('height', this.canvasHeight)
        .node() as HTMLCanvasElement;

    this.drawCross();
    this.drawBlockerAndTextAxis();
    this.drawCanvasLines();
    if (this.isIndividual) {
      this.mainG = this.svg.selectAll('g.main-g').data([null])
          .join('g')
          .attr('class', 'main-g')
          .attr('transform', `translate(${margin.left}, ${margin.top})`);
      this.drawShapelets();
    }
    if (this.enableZoom) {
      this.attachZoom();
    }
    this.applyExternalZoom();
  }

  public throttledRender = _.throttle(this.render, this.CANVAS_DEBOUNCE, {
    trailing: true,
  });
}

export default InstanceRenderer;
