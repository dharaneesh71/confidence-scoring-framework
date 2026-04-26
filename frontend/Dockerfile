# ============================================================
# CONFID.AI — Frontend Dockerfile  (multi-stage)
# Stage 1 — node:18-alpine  → builds the React app
# Stage 2 — nginx:alpine    → serves the static build
# Port: 80
#
# REACT_APP_API_URL is a build-time ARG so the same Dockerfile
# works for both local docker-compose and CS Launch production.
#
# Local dev default:   http://localhost:8000
# CS Launch production: pass --build-arg REACT_APP_API_URL=https://<your-backend-url>
# ============================================================

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# Install deps first (better layer caching — only re-runs if package.json changes)
COPY package*.json ./
RUN npm install --legacy-peer-deps

# Copy source and build
COPY . .

# Accept the backend URL at build time
ARG REACT_APP_API_URL=http://localhost:8000
ENV REACT_APP_API_URL=$REACT_APP_API_URL

RUN npm run build

# ── Stage 2: Serve ────────────────────────────────────────────────────────────
FROM nginx:alpine

# Copy the React production build
COPY --from=builder /app/build /usr/share/nginx/html

# Replace the default nginx config with our SPA-aware config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
