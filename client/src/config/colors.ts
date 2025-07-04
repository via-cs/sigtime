/*
 * colors.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-04 17:31:17
 * @modified: 2025-03-26 01:39:56
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import {blue, orange, red} from '@mui/material/colors';
import chroma from 'chroma-js';

export const colors = {
  pos: orange[800],
  posLight: chroma(orange[800]).alpha(0.2).hex(),
  neg: blue[700],
  negLight: chroma(blue[700]).alpha(0.2).hex(),
  highlight: chroma(red[700]).alpha(1).hex(),
  highlightLight: chroma(red[700]).alpha(0.6).hex(),
  highlightLight2: chroma(red[700]).alpha(0.2).hex(),
  shapes: (() => {
    const arr = chroma.scale('Set3').colors(10);
    [arr[1], arr[3]] = [arr[3], arr[1]];
    return arr;
  })(),
};
