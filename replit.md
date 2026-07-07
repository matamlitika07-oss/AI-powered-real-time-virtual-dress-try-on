# Virtual Try-On

An AI-powered real-time virtual dress try-on application. Uses MediaPipe Pose for body tracking in the browser and overlays garments on a live webcam feed.

## Architecture

PNPM monorepo with three main packages:

| Package | Path | Purpose |
|---|---|---|
| `@workspace/virtual-tryon` | `artifacts/virtual-tryon/` | React + Vite frontend with MediaPipe |
| `@workspace/api-server` | `artifacts/api-server/` | Node/Express REST API |
| `@workspace/db` | `lib/db/` | Drizzle ORM schema + migrations |
| `@workspace/api-spec` | `lib/api-spec/` | OpenAPI spec + Orval codegen |

## Running the project

Two workflows are configured in Replit:

- **Virtual Try-On** — Vite dev server on port 5173 (`BASE_PATH=/`)
- **API Server** — Express server on port 8080 (`/api`)

Start both from the Replit workflow panel, or use the Run button.

## Development commands

```bash
# Install dependencies
pnpm install

# Push DB schema changes
pnpm --filter @workspace/db run push

# Regenerate API hooks from OpenAPI spec
pnpm --filter @workspace/api-spec run codegen

# Build API server
pnpm --filter @workspace/api-server run build
```

## Environment

- `DATABASE_URL` — managed by Replit (PostgreSQL, pre-provisioned)
- `SESSION_SECRET` — stored as a Replit secret

## Tech stack

- **Frontend**: React 19, Vite 7, Tailwind CSS 4, Framer Motion, Radix UI, MediaPipe Pose
- **Backend**: Node.js 24, Express 5, Pino logging, Esbuild
- **Database**: PostgreSQL 16 via Drizzle ORM
- **Codegen**: Orval (OpenAPI → React Query hooks + Zod schemas)
