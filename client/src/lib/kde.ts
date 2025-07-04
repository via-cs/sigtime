/*
 * kde.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-03-21 16:28:45
 * @modified: 2025-03-28 15:01:53
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import * as d3 from 'd3';

export interface IKDE1dConfig {
  bandwidth: number;
  kernel: 'gaussian' | 'epanechnikov' | 'uniform';
  gridSize: number;
  max: number;
  min: number;
  boundary: number;
}

export class KDE1d {
  data: number[];
  config: IKDE1dConfig;

  constructor(data: number[], config?: Partial<IKDE1dConfig>) {
    this.data = data;
    this.config = {
      bandwidth: 0.07,
      kernel: 'gaussian',
      gridSize: 200,
      max: 1,
      min: 0,
      boundary: 0.05,
      ...config,
    };
  }

  /**
   *
   * @returns [threshold, density][]
   */
  public fit(normalized: boolean = false) {
    const {bandwidth, kernel, gridSize} = this.config;
    const thresholds = d3.range(this.config.min - this.config.boundary, this.config.max + this.config.boundary, (this.config.max - this.config.min + 2 * this.config.boundary) / gridSize);

    const kernelFunction = (u: number) => {
      switch (kernel) {
        case 'gaussian':
          return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * u * u);
        case 'epanechnikov':
          return Math.abs(u) <= 1 ? 0.75 * (1 - u * u) : 0;
        case 'uniform':
          return Math.abs(u) <= 1 ? 0.5 : 0;
        default:
          throw new Error(`Unknown kernel: ${kernel}`);
      }
    };

    const kde = (thresholds: number[], data: number[], bandwidth: number) => {
      const vals = thresholds.map((t) => {
        const sum = data.reduce((acc, d) => acc + kernelFunction((t - d) / bandwidth), 0);
        return [t, sum / (data.length * bandwidth)];
      });

      if (normalized) {
        const max = Math.max(...vals.map((v) => v[1]));
        return vals.map((v) => [v[0], v[1] / max]);
      }

      return vals;
    };

    return kde(thresholds, this.data, bandwidth);
  }
}
