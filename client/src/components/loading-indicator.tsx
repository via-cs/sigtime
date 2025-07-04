/*
 * loading-indicator.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-02-04 17:25:59
 * @modified: 2025-02-05 02:38:18
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import {CircularProgress} from '@mui/material';
import * as React from 'react';

export interface ILoadingIndicatorProps {
  size?: number;
  fullScreen?: boolean;
}

export function LoadingIndicator(props: ILoadingIndicatorProps) {
  if (props.fullScreen) {
    return (
      <div className={'absolute inset-0 bg-neutral-800 opacity-50 flex backdrop-blur-xs justify-center items-center z-50'}>
        <CircularProgress variant={'indeterminate'} sx={{color: 'white'}} size={props.size ?? 24} />
      </div>
    );
  } else {
    return (
      <CircularProgress variant={'indeterminate'} size={props.size ?? 16} />
    );
  }
}
