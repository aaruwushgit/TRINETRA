# Multi-stage, same reasoning as the backend image: nothing needed only to
# build should survive into the runtime layer.
#
# `NEXT_PUBLIC_*` values are inlined into the client bundle at BUILD time, not
# read at runtime, so they arrive as build args. That is a property of Next.js,
# and it is why the TomTom key is necessarily visible in the browser — see
# .env.example for how to restrict it.
FROM node:22-alpine AS deps
WORKDIR /app
COPY services/frontend/package.json services/frontend/package-lock.json* ./
# `npm ci` when a lockfile exists (reproducible), `npm install` on first build.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

FROM node:22-alpine AS builder
WORKDIR /app
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ARG NEXT_PUBLIC_TOMTOM_API_KEY=
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_TOMTOM_API_KEY=$NEXT_PUBLIC_TOMTOM_API_KEY
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY services/frontend/ ./
RUN npm run build

# ── runtime ─────────────────────────────────────────────────────────────────
FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

# Run as a non-root user: this container is the one exposed to a browser.
RUN addgroup -S -g 1001 nodejs && adduser -S -u 1001 nextjs

# `output: "standalone"` produces a server bundle with only the reachable
# dependencies, so this copies ~60 MB rather than a whole node_modules.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
