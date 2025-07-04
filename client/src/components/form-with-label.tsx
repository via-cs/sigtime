/*
 * form-with-label.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-27 18:31:08
 * @modified: 2025-02-04 17:08:55
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import config from '@/config';
import {Tooltip} from '@mui/material';
import * as React from 'react';

export interface IFormWithLabelProps {
  className?: string;
  label: string;
  labelWidth?: string;
  help?: string | React.ReactNode;
  children: React.ReactNode;
}

export function FormWithLabel(props: IFormWithLabelProps) {
  const content = (
    <div className={`flex flex-row gap-2 items-center ${props.className}`}>
      <div className={'flex-none'} style={{width: props.labelWidth ?? config.defaultLabelWidth}}>{props.label}</div>
      <div className={'flex-1'}>{props.children}</div>
    </div>
  );

  if (props.help) {
    return (
      <Tooltip title={props.help} arrow placement={'top'}>
        {content}
      </Tooltip>
    );
  }

  return content;
}
