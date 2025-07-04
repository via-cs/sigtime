/*
 * vis.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-05 02:28:29
 * @modified: 2025-02-05 02:37:30
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import * as d3 from 'd3';

/**
 * Calculate the relative transform between two zoom transforms.
 * @param transform - The current zoom transform.
 * @param lastTransform - The last zoom transform.
 * @returns The relative transform.
 */
export const relativeTransform = (transform: d3.ZoomTransform, lastTransform: d3.ZoomTransform) => {
  const kRelative = transform.k / lastTransform.k;
  const xRelative = transform.x - kRelative * lastTransform.x;
  const yRelative = transform.y - kRelative * lastTransform.y;
  return d3.zoomIdentity.translate(xRelative, yRelative).scale(kRelative);
};
