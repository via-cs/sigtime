import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import google from 'eslint-config-google';
import tseslint from 'typescript-eslint';
import unusedImports from 'eslint-plugin-unused-imports';

delete google.rules['valid-jsdoc'];
delete google.rules['require-jsdoc'];

export default tseslint.config(
    {ignores: ['dist']},
    google,
    {
      extends: [js.configs.recommended, ...tseslint.configs.recommended],
      files: ['**/*.{ts,tsx}'],
      languageOptions: {
        ecmaVersion: 2020,
        globals: globals.browser,
      },
      plugins: {
        'react-hooks': reactHooks,
        'react-refresh': reactRefresh,
        'unused-imports': unusedImports,
      },
      rules: {
        ...reactHooks.configs.recommended.rules,
        'react-refresh/only-export-components': [
          'warn',
          {allowConstantExport: true},
        ],
        '@typescript-eslint/no-unused-vars': 'warn',
        'unused-imports/no-unused-imports': 'warn',
        '@typescript-eslint/ban-ts-comment': 'off',
        'require-jsdoc': 'off',
        'max-len': 'off',
        'no-unused-vars': 'warn',
        'spaced-comment': 'warn',
        'react-hooks/rules-of-hooks': 'warn', // Checks rules for Hooks
        'react-hooks/exhaustive-deps': 'warn', // Checks effect dependencies
        'no-invalid-this': 'warn',
        '@typescript-eslint/no-explicit-any': 'off',
        '@typescript-eslint/no-empty-object-type': 'off',
        'camelcase': 'off',
        'spaced-comment': 'off',
      },
    },
);
