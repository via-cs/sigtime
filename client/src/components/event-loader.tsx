/*
 * event-bus.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-29 14:04:22
 * @modified: 2025-03-19 14:04:59
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import {windowWidthAtom} from '@/store/atom';
import {useAtom} from 'jotai';
import * as React from 'react';

export interface IEventBusProps {
}

export function EventLoader(props: IEventBusProps) {
  const [windowWidth, setWindowWidth] = useAtom(windowWidthAtom);

  React.useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    window.addEventListener('resize', handleResize);
    window.addEventListener('load', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('load', handleResize);
    };
  });

  return null;
}
