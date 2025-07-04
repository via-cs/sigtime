/*
 * parallel-coords-renderer.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-03-19 18:30:00
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import * as d3 from 'd3';
import {colors} from '@/config/colors';
import {IParams, ISelection, ITsDataInstances, IProcessedShapeData} from '@/types';

/**
 * Used as an alternative design to the ts-renderer. It is a plain parallel coordinates renderer.
 * It is not used in the current implementation.
 */
class ParallelCoordsRenderer {
  svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>;
  width: number;
  height: number;

  margin = {top: 60, right: 30, bottom: 20, left: 30};
  canvasWidth: number;
  canvasHeight: number;

  allInstances: ITsDataInstances[] = [];
  shapelets: IProcessedShapeData[] = [];
  params: IParams;
  selection: ISelection;

  hasConfig: boolean = false;
  container: d3.Selection<SVGGElement, unknown, HTMLElement, unknown>;

  // Scales for each dimension (shapelet)
  dimensions: string[] = [];
  y: { [key: string]: d3.ScaleLinear<number, number, never> } = {};
  x: d3.ScalePoint<string>;

  // For scrolling when there are many dimensions
  scrollContainer: d3.Selection<SVGGElement, unknown, HTMLElement, unknown>;
  scrollWidth: number;
  isScrollable: boolean = false;
  svgWrapper: HTMLDivElement | null = null;

  constructor(svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown> | SVGSVGElement) {
    this.svg = svg instanceof SVGSVGElement ? d3.select(svg) : svg;

    // Create a wrapper div for SVG to enable horizontal scrolling
    const svgNode = this.svg.node();
    if (svgNode && svgNode.parentNode) {
      // Create wrapper div for overflow scrolling
      this.svgWrapper = document.createElement('div');
      this.svgWrapper.style.width = '100%';
      this.svgWrapper.style.height = '100%';
      this.svgWrapper.style.overflowX = 'auto';
      this.svgWrapper.style.overflowY = 'hidden';
      this.svgWrapper.id = 'parallel-coords-wrapper';

      // Replace SVG with wrapper + SVG
      svgNode.parentNode.replaceChild(this.svgWrapper, svgNode);
      this.svgWrapper.appendChild(svgNode);
    }

    this.width = this.svg.node().clientWidth;
    this.height = this.svg.node().clientHeight;
    this.canvasWidth = this.width - this.margin.left - this.margin.right;
    this.canvasHeight = this.height - this.margin.top - this.margin.bottom;
  }

  public destroy() {
    this.svg.selectAll('*').remove();

    // Restore original DOM structure if wrapper was created
    if (this.svgWrapper) {
      const svgNode = this.svg.node();
      if (svgNode && this.svgWrapper.parentNode) {
        this.svgWrapper.parentNode.replaceChild(svgNode, this.svgWrapper);
        this.svgWrapper = null;
      }
    }
  }

  public setRenderConfig(config: {
    allInstances: ITsDataInstances[];
    shapelets: IProcessedShapeData[];
    params: IParams;
    selection: ISelection;
    margin?: {top: number, right: number, bottom: number, left: number};
  }) {
    this.params = config.params;
    this.selection = config.selection;
    this.margin = config.margin || this.margin;
    this.hasConfig = true;
    this.allInstances = config.allInstances;
    this.shapelets = config.shapelets;

    this.canvasWidth = this.width - this.margin.left - this.margin.right;
    this.canvasHeight = this.height - this.margin.top - this.margin.bottom;
  }

  public render() {
    if (!this.hasConfig) {
      console.error('Rendering config not set');
      return;
    }

    this.destroy();

    // Create dimensions array - one for each shapelet
    this.dimensions = this.shapelets.map((s) => `shapelet-${s.id}`);

    // Calculate total width needed based on dimensions
    const dimensionWidth = 65; // Width allocated for each dimension
    this.scrollWidth = this.dimensions.length * dimensionWidth;
    this.isScrollable = this.scrollWidth > this.canvasWidth;

    // Set SVG width based on whether scrolling is needed
    if (this.isScrollable) {
      // Set the SVG width to accommodate all dimensions
      this.svg.attr('width', this.scrollWidth + this.margin.left + this.margin.right);

      // Ensure the wrapper has scrolling enabled
      if (this.svgWrapper) {
        this.svgWrapper.style.overflowX = 'auto';
      }
    } else {
      // If no scrolling needed, set SVG width to container width
      this.svg.attr('width', '100%');

      // Disable scrolling on wrapper
      if (this.svgWrapper) {
        this.svgWrapper.style.overflowX = 'hidden';
      }
    }

    // Create container
    this.container = this.svg.append('g')
        .attr('transform', `translate(${this.margin.left},${this.margin.top})`)
        .attr('id', 'parallel-coords-container');

    this.scrollContainer = this.container;

    // Create a scale for x axis - distributing dimensions
    this.x = d3.scalePoint()
        .domain(this.dimensions)
        .range(this.isScrollable ?
            [0, this.scrollWidth] : // Use full scrollWidth when scrollable
            [0, this.canvasWidth], // Use canvasWidth when not scrollable
        )
        .padding(0.1);

    // Create scales for y axis - one for each dimension
    this.dimensions.forEach((dimension, i) => {
      // Y scale for each dimension - distance to shapelet
      // Swapped domain to set top to 0 (closest) and bottom to 1 (farthest)
      this.y[dimension] = d3.scaleLinear()
          .domain([0, 1])
          .range([0, this.canvasHeight]);
    });

    // Filter instances based on selection if needed
    const instances = this.filterInstances();

    // Draw axes
    this.drawAxes();

    // Draw thumbnails and labels
    this.drawThumbnails();

    // Draw lines for each instance
    this.drawLines(instances);
  }

  private filterInstances(): ITsDataInstances[] {
    // Filter instances based on label enablement
    let instances = this.allInstances.filter((d) =>
      (d.pos && this.params.posEnabled) || (!d.pos && this.params.negEnabled),
    );

    // Filter by selection if shapelets are selected and filter threshold is set
    if (this.selection.shapelets && this.selection.shapelets.length > 0) {
      instances = instances.filter((instance) =>
        this.selection.shapelets.every((shapeletIdx) =>
          instance.tf[shapeletIdx] <= this.selection.filterThereshold,
        ),
      );
    }

    return instances;
  }

  private drawAxes() {
    // Draw axes for each dimension
    this.dimensions.forEach((dimension, i) => {
      const shapeletId = parseInt(dimension.replace('shapelet-', ''));
      const g = this.scrollContainer.append('g')
          .attr('transform', `translate(${this.x(dimension)},0)`)
          .attr('id', `axis-group-${shapeletId}`);

      // Draw axis line with gray color and 1.5px width
      g.append('line')
          .attr('y1', 0)
          .attr('y2', this.canvasHeight)
          .attr('stroke', '#ccc')
          .attr('stroke-width', 1.5)
          .attr('id', `axis-line-${shapeletId}`);

      // Add highlight for selected shapelets
      const isSelected = this.selection.shapelets?.includes(shapeletId);

      if (isSelected) {
        const rectWidth = 10;
        // Highlight the axis with a light red background
        g.append('rect')
            .attr('x', -rectWidth / 2)
            .attr('y', 0)
            .attr('width', rectWidth)
            .attr('height', this.canvasHeight * this.selection.filterThereshold)
            .attr('fill', colors.highlightLight)
            .attr('opacity', 0.5)
            .attr('id', `axis-highlight-${shapeletId}`);
      }
    });
  }

  private drawThumbnails() {
    // Draw thumbnail previews and labels for each axis
    this.dimensions.forEach((dimension, i) => {
      const shapeletId = parseInt(dimension.replace('shapelet-', ''));
      const shapelet = this.shapelets.find((s) => s.id === shapeletId);

      if (shapelet) {
        const thumbnailHeight = 30;
        const thumbnailWidth = 65;
        const labelHeight = 15;
        const totalHeight = thumbnailHeight + labelHeight;

        const xPos = this.x(dimension) - thumbnailWidth / 2;
        const yPos = -totalHeight;

        const thumbnailContainer = this.scrollContainer.append('g')
            .attr('transform', `translate(${xPos},${yPos})`)
            .attr('id', `thumbnail-container-${shapeletId}`);

        // Outer container with gray border
        thumbnailContainer.append('rect')
            .attr('width', thumbnailWidth)
            .attr('height', totalHeight)
            .attr('rx', 2)
            .attr('ry', 2)
            .attr('stroke', '#ccc')
            .attr('stroke-width', 1)
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

        // Add label
        thumbnailContainer.append('text')
            .attr('x', thumbnailWidth / 2)
            .attr('y', thumbnailHeight + labelHeight / 2 + 5)
            .attr('text-anchor', 'middle')
            .attr('fill', '#333')
            .attr('class', 'text-mono-gray')
            .text(`S${shapelet.id + 1}`)
            .attr('id', `thumbnail-label-${shapeletId}`);
      }
    });
  }

  private drawLines(instances: ITsDataInstances[]) {
    // Create a group for all lines
    const linesGroup = this.scrollContainer.append('g')
        .attr('id', 'parallel-coords-lines');

    // Calculate opacity based on number of instances
    const alpha = Math.max(this.params.lineOpacity, 2 / instances.length);

    // Create straight line generator
    const line = d3.line<[string, number]>()
        .x((d) => this.x(d[0]))
        .y((d) => this.y[d[0]](d[1]));

    // Draw lines for each instance
    instances.forEach((instance) => {
      // Create array of [dimension, value] pairs for this instance
      const values: [string, number][] = this.dimensions.map((dim) => {
        const shapeletId = parseInt(dim.replace('shapelet-', ''));
        return [dim, instance.tf[shapeletId]];
      });

      // Draw the line
      linesGroup.append('path')
          .datum(values)
          .attr('fill', 'none')
          .attr('stroke', instance.pos ? colors.pos : colors.neg)
          .attr('stroke-width', 1.5)
          .attr('stroke-opacity', alpha)
          .attr('d', line)
          .attr('id', `instance-line-${instance.iid}`);
    });
  }

  public resize(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.canvasWidth = this.width - this.margin.left - this.margin.right;
    this.canvasHeight = this.height - this.margin.top - this.margin.bottom;

    if (this.hasConfig) {
      this.render();
    }
  }
}

export default ParallelCoordsRenderer;
