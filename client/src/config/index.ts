/*
 * index.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-27 18:37:51
 * @modified: 2025-03-26 20:32:44
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import {CurveType} from '@/types';

const config = {
  defaultLabelWidth: '6em',
  apiRoot: '/api',
  defaultCurveType: 'linear' as CurveType,
  maxShapelets: 10,
  maxDetailedInstances: 500,
};

export default config;
