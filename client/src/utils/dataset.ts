/*
 * dataset.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-04 20:25:26
 * @modified: 2025-02-04 20:57:04
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import {datasetConfig} from '@/config/datasets';

export const getDsConfig = (dataset: string) => datasetConfig.find((d) => d.name === dataset);
