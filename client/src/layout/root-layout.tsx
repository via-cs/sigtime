/*
 * root-layout.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-27 17:55:49
 * @modified: 2025-03-19 16:41:10
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import PanelDetail from '@/layout/panel-detail';
import {PanelOverview} from '@/layout/panel-overview';
import PanelTs from '@/layout/panel-ts';
import * as React from 'react';

export interface IRootLayoutProps {
}

export function RootLayout(props: IRootLayoutProps) {
  return (
    <div className={'w-full h-full grid grid-cols-24 grid-rows-12 gap-2 p-2'}>
      <div className={`panel col-span-12 row-span-12`}>
        <PanelOverview />
      </div>

      <div className={'panel col-span-12 row-span-8'}>
        <PanelTs />
      </div>

      <div className={'panel col-span-12 row-span-4'}>
        <PanelDetail />
      </div>

    </div>
  );
}
