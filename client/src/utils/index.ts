/*
 * index.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-04 17:19:10
 * @modified: 2025-03-26 20:16:11
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import chroma from 'chroma-js';
import {ColorRGBA} from 'webgl-plot';
import {clsx, type ClassValue} from 'clsx';
import {twMerge} from 'tailwind-merge';

/**
 * Check if two values are likely equal. Converted to string and compare.
 * @param a - The first value.
 * @param b - The second value.
 * @returns True if the values are likely equal, false otherwise.
 */
export const likelyEq = (a: number | string, b: number | string): boolean => {
  if (a === b) return true;
  if (String(a) === String(b)) return true;
  return false;
};


export const lorem = (len: number): string => {
  const words = ['lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit'];
  let result = '';
  while (result.length < len) {
    const word = words[Math.floor(Math.random() * words.length)];
    if (result.length + word.length + 1 <= len) {
      result += (result ? ' ' : '') + word;
    } else {
      break;
    }
  }
  return result;
};


export const chromaToColorRGBA = (c: chroma.Color) => {
  return new ColorRGBA(c.get('rgb.r'), c.get('rgb.g'), c.get('rgb.b'), c.get('alpha'));
};


export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}


export function toggleArrayItem<T>(array: T[], item: T) {
  return array.includes(item) ? array.filter((i) => i !== item) : [...array, item];
}


export function objEquals(a: Record<string, any>, b: Record<string, any>) {
  return JSON.stringify(a || {}) === JSON.stringify(b || {});
}
