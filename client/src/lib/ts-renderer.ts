/*
 * ts-renderer.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-03-21 15:58:51
 * @modified: 2025-03-28 15:03:20
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import {colors} from '@/config/colors';
import {KDE1d} from '@/lib/kde';
import {IParams, IProcessedShapeData, ISelection, ITsDataInstances, AtomSetter, ISortConfig} from '@/types';
import * as d3 from 'd3';

type Margin = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

export type TsRendererMode = 'individual' | 'overview';
class TsRenderer {
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>;
  width: number;
  height: number;

  margin?: Margin;

  hasConfig: boolean;
  mode: TsRendererMode;
  allInstances: ITsDataInstances[] = [];
  shapelets: IProcessedShapeData[] = [];
  params: IParams;
  selection: ISelection;
  setFilteredInstances?: AtomSetter<ITsDataInstances[]>;

  onClickShapelet?: (shapeletIdx: number, mode: TsRendererMode) => void;

  x: d3.ScalePoint<string>;
  overviewY: d3.ScaleLinear<number, number, never>;
  dimensions: string[];

  SHAPELET_WIDTH = 70;
  SHAPELET_HEIGHT = 30;
  LABEL_HEIGHT = 15;
  CIRCLE_RADIUS = 4;
  CIRCLE_OPACITY = 0.2;
  BOUNDARY = 0.05;

  DETAILED_MARGIN_LEFT = 240;
  DETAILED_MARGIN_RIGHT = 40;
  DETAILED_SHAPELET_WIDTH = 60;

  mainG: d3.Selection<SVGGElement, unknown, HTMLElement, unknown>;
  filteredInstances: ITsDataInstances[];
  sort: ISortConfig;

  constructor(svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown> | SVGSVGElement, setters: {
    setFilteredInstances?: AtomSetter<ITsDataInstances[]>;
  }) {
    this.svg = svg instanceof SVGSVGElement ? d3.select(svg) : svg;
    this.width = this.svg.node().clientWidth;
    this.height = this.svg.node().clientHeight;
    this.setFilteredInstances = setters.setFilteredInstances;
  }

  public setRenderConfig(config: {
    mode: TsRendererMode;
    allInstances: ITsDataInstances[];
    params: IParams;
    selection: ISelection;
    shapelets: IProcessedShapeData[];
    sort: ISortConfig;
  }) {
    this.mode = config.mode;
    this.params = config.params;
    this.selection = config.selection;
    this.shapelets = config.shapelets;
    this.allInstances = config.allInstances;
    this.sort = config.sort;
    this.hasConfig = true;

    if (this.mode === 'overview') {
      this.SHAPELET_WIDTH = 70;
      this.margin = {
        top: 50, right: 50, bottom: 10, left: 60,
      };
    } else {
      this.SHAPELET_WIDTH = this.DETAILED_SHAPELET_WIDTH;
      this.margin = {
        top: 50, right: this.DETAILED_MARGIN_RIGHT, bottom: 10, left: this.DETAILED_MARGIN_LEFT,
      };
    }

    this.filteredInstances = this.filterInstances();
  }


  private get isFiltered() {
    return this.selection.shapelets && this.selection.shapelets.length > 0;
  }

  public filterInstances() {
    if (!this.hasConfig) {
      return [];
    }
    let instances = this.allInstances.filter((d) =>
      (d.pos && this.params.posEnabled) || (!d.pos && this.params.negEnabled),
    );

    if (this.selection.shapelets && this.selection.shapelets.length > 0) {
      if (this.selection.filterMode === 'intersect') {
        instances = instances.filter((instance) =>
          this.selection.shapelets.every((shapeletIdx) =>
            instance.tf[shapeletIdx] <= this.selection.filterThereshold,
          ),
        );
      } else {
        instances = instances.filter((instance) =>
          this.selection.shapelets.some((shapeletIdx) =>
            instance.tf[shapeletIdx] <= this.selection.filterThereshold,
          ),
        );
      }
    }
    this.setFilteredInstances?.(instances);
    return instances;
  }

  private drawThumbnails() {
    // Draw thumbnail previews and labels for each axis
    this.dimensions.forEach((dimension, i) => {
      const shapeletId = parseInt(dimension.replace('shapelet-', ''));
      const shapelet = this.shapelets[shapeletId];

      if (shapelet) {
        const thumbnailHeight = this.SHAPELET_HEIGHT;
        const thumbnailWidth = this.SHAPELET_WIDTH;
        const labelHeight = this.LABEL_HEIGHT;
        const totalHeight = thumbnailHeight + labelHeight;

        const xPos = this.x(dimension) - thumbnailWidth / 2 + this.margin.left;
        const yPos = this.margin.top - totalHeight;

        const thumbnailContainer = this.svg.append('g')
            .attr('transform', `translate(${xPos},${yPos})`)
            .attr('id', `thumbnail-container-${shapeletId}`)
            .attr('cursor', 'pointer')
            .on('click', () => {
              this.onClickShapelet?.(i, this.mode);
            });

        const isThisShapeletFiltered = this.isFiltered && this.selection.shapelets.includes(shapeletId);
        // Outer container with gray border
        thumbnailContainer.append('rect')
            .attr('width', thumbnailWidth)
            .attr('height', totalHeight)
            .attr('rx', 3)
            .attr('ry', 3)
            .attr('stroke', isThisShapeletFiltered ? colors.highlightLight : '#ccc')
            .attr('stroke-width', isThisShapeletFiltered ? 2 : 1)
            .attr('fill-opacity', 0.2)
            .attr('fill', 'white')
            .attr('id', `thumbnail-border-${shapeletId}`);

        // Draw thumbnail background
        thumbnailContainer.append('rect')
            .attr('y', 0)
            .attr('width', thumbnailWidth)
            .attr('height', thumbnailHeight)
            .attr('fill', 'none')
            .attr('id', `thumbnail-bg-${shapeletId}`);

        // Draw shapelet pattern using proper color
        if (shapelet.vals && shapelet.vals.length > 0) {
          const xScale = d3.scaleLinear()
              .domain([0, shapelet.vals.length - 1])
              .range([5, thumbnailWidth - 5]);

          const yScale = d3.scaleLinear()
              .domain([d3.min(shapelet.vals), d3.max(shapelet.vals)])
              .range([thumbnailHeight - 5, 5]);

          const line = d3.line<number>()
              .x((d, i) => xScale(i))
              .y((d) => yScale(d))
              .curve(d3.curveMonotoneX);

          thumbnailContainer.append('path')
              .datum(shapelet.vals)
              .attr('fill', 'none')
              .attr('stroke', colors.shapes[shapelet.id % colors.shapes.length])
              .attr('stroke-width', 2)
              .attr('d', line)
              .attr('id', `thumbnail-path-${shapeletId}`);
        }

        thumbnailContainer.append('text')
            .attr('x', thumbnailWidth / 2)
            .attr('y', thumbnailHeight + labelHeight / 2 + 5)
            .attr('text-anchor', 'middle')
            .attr('fill', '#333')
            .attr('class', 'text-mono-gray')
            .text(`S${shapelet.id + 1}`)
            .attr('id', `thumbnail-label-${shapeletId}`);

        console.log(this.sort);

        if (this.sort.key === `shapelet` && this.sort.sid === shapeletId) {
          thumbnailContainer.append('text')
              .attr('x', thumbnailWidth - 10)
              .attr('y', thumbnailHeight + labelHeight / 2 + 5)
              .attr('text-anchor', 'middle')
              .attr('fill', '#333')
              .attr('class', 'text-mono-gray')
              .text(this.sort.order === 'asc' ? '↑' : '↓')
              .attr('id', `thumbnail-label-${shapeletId}`);
        }
      }
    });
  }

  public drawDimensions(withKDE: boolean = true) {
    // dimension lines
    this.mainG.selectAll('line.dimension-line')
        .data(this.dimensions)
        .join('line')
        .attr('class', 'dimension-line')
        .attr('x1', (d) => this.x(d))
        .attr('y1', 0)
        .attr('x2', (d) => this.x(d))
        .attr('y2', this.overviewY(1 + this.BOUNDARY))
        .attr('stroke', '#ccc')
        .attr('stroke-width', 1.5);

    const densities = [];
    let densityNegMax = -Infinity;
    let densityPosMax = -Infinity;
    for (let di = 0; di < this.dimensions.length; di++) {
      const dimensionTfsNeg = this.filteredInstances
          .filter((d) => !d.pos)
          .map((d) => d.tf[di]);
      const dimensionTfsPos = this.filteredInstances
          .filter((d) => d.pos)
          .map((d) => d.tf[di]);

      const kdeNeg = new KDE1d(dimensionTfsNeg, {
        bandwidth: 0.011,
      });
      const kdePos = new KDE1d(dimensionTfsPos, {
        bandwidth: 0.011,
      });

      const densityNeg = kdeNeg.fit(false);
      const densityPos = kdePos.fit(false);

      densities.push({
        neg: densityNeg,
        pos: densityPos,
      });
      densityNegMax = Math.max(densityNegMax, Math.max(...densityNeg.map((d) => d[1])));
      densityPosMax = Math.max(densityPosMax, Math.max(...densityPos.map((d) => d[1])));
    }

    densities.forEach((d) => {
      d.neg = d.neg.map((d) => [d[0], d[1] / densityNegMax]);
      d.pos = d.pos.map((d) => [d[0], d[1] / densityPosMax]);
    });

    for (let di = 0; di < this.dimensions.length; di++) {
      const xScale = d3.scaleLinear()
          .domain([0, 1])
          .range([0, (this.width - this.margin.left - this.margin.right) / this.dimensions.length / 2 * 1.1]);

      if (withKDE) {
        const areaNeg = d3.area<[number, number]>()
            .y((d) => this.overviewY(d[0]))
            .x1((d) => this.x(this.dimensions[di]) + xScale(d[1]))
            .x0((d) => this.x(this.dimensions[di]) - xScale(d[1]));

        const areaPos = d3.area<[number, number]>()
            .y((d) => this.overviewY(d[0]))
            .x1((d) => this.x(this.dimensions[di]) + xScale(d[1]))
            .x0((d) => this.x(this.dimensions[di]) - xScale(d[1]));

        this.mainG.append('path')
            .attr('id', `density-neg-${di}`)
            .attr('class', 'density-line-neg')
            .datum(densities[di].neg)
            .attr('fill', colors.neg)
            .attr('fill-opacity', 0.2)
            .attr('stroke', colors.neg)
            .attr('stroke-opacity', 0.3)
        // @ts-ignore
            .attr('d', areaNeg);

        this.mainG.append('path')
            .attr('id', `density-pos-${di}`)
            .attr('class', 'density-line-pos')
            .datum(densities[di].pos)
            .attr('fill', colors.pos)
            .attr('fill-opacity', 0.2)
            .attr('stroke', colors.pos)
            .attr('stroke-opacity', 0.3)
        // @ts-ignore
            .attr('d', areaPos);
      }
    }
  }

  private drawInstances() {
    this.filteredInstances.forEach((instance) => {
      instance.tf.forEach((tf, i) => {
        const x = this.x(this.dimensions[i]);
        const y = this.overviewY(tf);
        this.mainG.append('circle')
            .attr('cx', x)
            .attr('cy', y)
            .attr('r', this.CIRCLE_RADIUS)
            .attr('fill', instance.pos ? colors.pos : colors.neg)
            .attr('fill-opacity', this.CIRCLE_OPACITY);
      });
    });
  }

  private drawFilterStatus() {
    if (!this.isFiltered) {
      return;
    }

    // draw a red bold line on the dimension with filitering turned on
    this.selection.shapelets.forEach((shapeletIdx) => {
      const shapelet = this.shapelets[shapeletIdx];
      const x = this.x(this.dimensions[shapeletIdx]);
      this.mainG.append('line')
          .attr('x1', x)
          .attr('y1', 0)
          .attr('x2', x)
          .attr('y2', this.overviewY(this.selection.filterThereshold))
          .attr('stroke', colors.highlight)
          .attr('stroke-opacity', 0.1)
          .attr('stroke-width', this.SHAPELET_WIDTH);
    });
  }

  private drawConnections() {
    // parallel coornidates lines connecting the instances
    const lines = this.filteredInstances.map((instance, index) => {
      const points = this.dimensions.map((dimension) => {
        const x = this.x(dimension);
        const y = this.overviewY(instance.tf[parseInt(dimension.replace('shapelet-', ''))]);
        return {
          x: x,
          y: y,
        };
      });
      return {
        points,
        pos: instance.pos,
        index,
        isSelected: this.selection.selectedInstance === index,
      };
    });

    this.mainG.selectAll('path.connection')
        .data(lines)
        .join('path')
        .attr('class', 'connection')
        .attr('stroke', (d) => d.isSelected ? colors.highlight : (d.pos ? colors.pos : colors.neg))
        .attr('d', (d) => d3.line<{x: number, y: number, pos: boolean}>()
            .x((d) => d.x)
            // @ts-ignore
            .y((d) => d.y)(d.points))
        .attr('stroke-opacity', (d) => d.isSelected ? 0.8 : 0.15)
        .attr('stroke-width', (d) => d.isSelected ? 3 : 1.5)
        .attr('fill', 'none');
  }

  private drawHoverStatus() {
    // add a horizon line at the y position of the hovered location to display the distance (0 - 1)
    // Create a horizontal line to show distance value on hover
    const hoverLine = this.mainG.append('line')
        .attr('class', 'hover-line')
        .attr('x1', 0)
        .attr('x2', this.width - this.margin.left - this.margin.right)
        .attr('stroke', colors.highlight)
        // .attr('stroke-dasharray', '3,3')
        .attr('stroke-width', 1.5)
        .attr('display', 'none');

    // Create text label to display the distance value
    const hoverText = this.mainG.append('text')
        .attr('class', 'hover-text')
        .attr('text-anchor', 'start')
        .attr('font-size', '10px')
        .attr('fill', colors.highlight)
        .attr('display', 'none');


    // Add mouse event listeners to the main group
    this.mainG.on('mousemove', (event) => {
      const [, mouseY] = d3.pointer(event);
      const distanceValue = this.overviewY.invert(mouseY);

      // Only show if within valid range
      if (distanceValue >= 0 && distanceValue <= 1) {
        // Update line position
        hoverLine
            .attr('y1', mouseY)
            .attr('y2', mouseY)
            .attr('display', null);

        // Update text with formatted distance value
        hoverText
            .attr('x', 5)
            .attr('y', mouseY - 5)
            .text(`Relevance: ${((1 - distanceValue) * 100).toFixed(2)}%`)
            .attr('display', null);
      } else {
        // Hide elements when outside valid range
        hoverLine.attr('display', 'none');
        hoverText.attr('display', 'none');
      }
    });

    // Hide elements when mouse leaves
    this.mainG.on('mouseleave', () => {
      hoverLine.attr('display', 'none');
      hoverText.attr('display', 'none');
    });
  }

  public render() {
    if (!this.hasConfig) {
      console.error('Rendering config not set');
      return;
    }

    this.destroy();
    this.dimensions = this.shapelets.map((s) => `shapelet-${s.id}`);
    this.overviewY = d3.scaleLinear()
        .domain([0 - this.BOUNDARY, 1 + this.BOUNDARY])
        .range([0, this.height - this.margin.top - this.margin.bottom]);
    this.x = d3.scalePoint()
        .domain(this.dimensions)
        .range([0, this.width - this.margin.left - this.margin.right]);
    this.mainG = this.svg.append('g')
        .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

    this.mainG.append('rect')
        .attr('width', this.width)
        .attr('height', this.height)
        .attr('fill', 'white')
        .attr('fill-opacity', 0.5);

    if (this.mode === 'overview') {
      this.drawThumbnails();
      this.drawDimensions(true);
      if (this.params.showConnections) {
        this.drawConnections();
      }
      this.drawInstances();
      this.drawFilterStatus();
      this.drawHoverStatus();
    } else {
      this.drawThumbnails();
      this.drawDimensions(false);
    }
  }

  public destroy() {
    this.svg.selectAll('*').remove();
  }
}


export default TsRenderer;
