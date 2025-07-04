FROM node:20-slim

WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && corepack prepare pnpm@9.9.0 --activate
RUN pnpm install

COPY . .

CMD ["npm", "run", "dev"]