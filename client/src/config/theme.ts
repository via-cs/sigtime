/*
 * theme.ts
 *
 * @project: ts-signature
 * @author: Juntong Chen (dev@jtchen.io)
 * @created: 2025-01-27 17:38:33
 * @modified: 2025-02-04 22:31:34
 *
 * Copyright (c) 2025 Juntong Chen. All rights reserved.
 */

import {colors} from '@/config/colors';
import {createTheme} from '@mui/material';

export const muiTheme = createTheme({
  typography: {
    fontFamily: 'Inter, Arial, sans-serif',
    fontSize: 14,
  },
  palette: {
    primary: {
      main: '#333333',
    },
    info: {
      main: colors.pos,
    },
    warning: {
      main: colors.neg,
    },
  },
  components: {
    MuiButton: {
      defaultProps: {
        size: 'small',
      },
    },
    MuiTextField: {
      defaultProps: {
        size: 'small',
      },
    },
    MuiSelect: {
      defaultProps: {
        variant: 'standard',
        size: 'small',
      },
    },
    MuiRadio: {
      defaultProps: {
        size: 'small',
      },
    },
    MuiCheckbox: {
      defaultProps: {
        size: 'small',
      },
    },
    MuiSlider: {
      defaultProps: {
        size: 'small',
      },
    },
  },
});
