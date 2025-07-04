/*
 * App.tsx
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-27 16:35:35
 * @modified: 2025-02-20 14:16:37
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */
import '@/styles/main.css';
import {AppBar, ThemeProvider} from '@mui/material';
import {muiTheme} from '@/config/theme';
import {RootLayout} from '@/layout/root-layout';
import {SWRConfig} from 'swr';
import {EventLoader} from '@/components/event-loader';

function App() {
  return (
    <ThemeProvider theme={muiTheme}>
      <EventLoader />
      <SWRConfig value={{
        refreshInterval: 0,
        // disable revalidation since backend data is almost static， should only change when key updates
        revalidateOnFocus: false,
        revalidateOnReconnect: false,
        revalidateOnMount: true,
        fetcher: (resource, init) => fetch(resource, init).then((res) => res.json()),
      }}>
        <div className={'w-screen h-screen flex flex-col'}>
          <AppBar position={'static'}>
            <div className={'flex flex-row w-full gap-2 px-3 py-2.5 items-center'}>
              <div className={'flex font-bold text-lg'}>SigTime</div>
              <div className={'flex-1'} />
              {/* <div className={'font-mono text-sm'}>{process.env.VITE_APP_BUILD_TIME}</div> */}
            </div>
          </AppBar>
          <div className={'flex flex-1 items-center justify-center min-h-0'}>
            <RootLayout />
          </div>
        </div>
      </SWRConfig>
    </ThemeProvider>
  );
}

export default App;
