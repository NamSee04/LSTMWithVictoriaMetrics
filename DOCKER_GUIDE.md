# Docker Setup Guide for LSTM Anomaly Detection

## Prerequisites

Install Docker and Docker Compose:
- **macOS**: [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- **Linux**: `sudo apt-get install docker.io docker-compose`
- **Windows**: [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)

## Step-by-Step Guide

### Step 1: Verify Your Files

Make sure you have these files:
```
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── .dockerignore
├── config.yaml
├── requirements.txt
└── src/
```

### Step 2: Update config.yaml for Docker

Update your `config.yaml` to use Docker service names:

```yaml
reader:
  datasource_url: "http://victoriametrics:8428"  # Use service name
  # ... rest of config

writer:
  datasource_url: "http://victoriametrics:8428"  # Use service name
  # ... rest of config
```

### Step 3: Build the Docker Image

```bash
# Build the application image
docker build -t lstm-anomaly:latest .

# Or let docker-compose build it
docker-compose build
```

### Step 4: Start All Services

```bash
# Start all services (VictoriaMetrics, Node Exporter, LSTM app)
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f lstm-anomaly
```

### Step 5: Verify Services Are Running

```bash
# Check all containers
docker-compose ps

# Should show 4 containers running:
# - victoriametrics (port 8428)
# - node-exporter (port 9100)
# - vmagent (port 8429)
# - lstm-anomaly
```

### Step 6: Access VictoriaMetrics UI

Open browser and go to:
- VictoriaMetrics: http://localhost:8428
- Node Exporter metrics: http://localhost:9100/metrics
- VMAgent: http://localhost:8429

### Step 7: Query Metrics

Test if metrics are being collected:
```bash
# Query CPU usage
curl 'http://localhost:8428/api/v1/query?query=node_cpu_seconds_total'

# Query anomaly metrics (after LSTM runs)
curl 'http://localhost:8428/api/v1/query?query=lstm_anomaly_score'
```

## Common Commands

### Start/Stop Services

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v

# Restart a specific service
docker-compose restart lstm-anomaly
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f lstm-anomaly

# Last 100 lines
docker-compose logs --tail=100 lstm-anomaly
```

### Execute Commands in Container

```bash
# Open shell in container
docker-compose exec lstm-anomaly /bin/bash

# Run Python commands
docker-compose exec lstm-anomaly python -c "import torch; print(torch.__version__)"

# Check config
docker-compose exec lstm-anomaly cat config.yaml
```

### Build and Update

```bash
# Rebuild after code changes
docker-compose build lstm-anomaly

# Rebuild and restart
docker-compose up -d --build lstm-anomaly

# Pull latest base images
docker-compose pull
```

## Running Just the LSTM App (Standalone)

If you have VictoriaMetrics running elsewhere:

### Option 1: Docker Run

```bash
# Build
docker build -t lstm-anomaly:latest .

# Run (update datasource_url in config.yaml first)
docker run -d \
  --name lstm-anomaly \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/model_checkpoints:/app/model_checkpoints \
  lstm-anomaly:latest
```

### Option 2: Custom Environment Variables

```bash
docker run -d \
  --name lstm-anomaly \
  -e VM_URL="http://your-vm-server:8428" \
  -v $(pwd)/config.yaml:/app/config.yaml \
  lstm-anomaly:latest
```

## Development Workflow

### Method 1: Mount Source Code (Live Reload)

For development, mount your source code:

```yaml
# Add to docker-compose.yml under lstm-anomaly service:
volumes:
  - ./src:/app/src:ro  # Read-only mount
  - ./config.yaml:/app/config.yaml
```

Then restart:
```bash
docker-compose restart lstm-anomaly
```

### Method 2: Rebuild on Changes

```bash
# Make code changes, then:
docker-compose build lstm-anomaly
docker-compose up -d lstm-anomaly
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs lstm-anomaly

# Check if config is valid
docker-compose exec lstm-anomaly cat /app/config.yaml

# Verify Python can import modules
docker-compose exec lstm-anomaly python -c "from src.main import main"
```

### Can't connect to VictoriaMetrics

```bash
# From inside container
docker-compose exec lstm-anomaly curl http://victoriametrics:8428/api/v1/query?query=up

# Check network
docker-compose exec lstm-anomaly ping victoriametrics
```

### Permission issues

```bash
# Fix ownership of model_checkpoints
sudo chown -R 1000:1000 model_checkpoints/
```

### Out of memory

```bash
# Check container resources
docker stats

# Add memory limits in docker-compose.yml:
services:
  lstm-anomaly:
    mem_limit: 2g
    memswap_limit: 2g
```

## Production Considerations

### 1. Use specific versions

```dockerfile
FROM python:3.11.8-slim  # Not :latest
```

### 2. Health checks

Add to docker-compose.yml:
```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8428')"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### 3. Resource limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### 4. Secrets management

Don't hardcode credentials. Use Docker secrets or environment files:

```bash
# Create .env file
echo "VM_URL=http://victoriametrics:8428" > .env

# Reference in docker-compose.yml
env_file:
  - .env
```

## Clean Up

```bash
# Stop and remove containers, networks
docker-compose down

# Also remove volumes
docker-compose down -v

# Remove images
docker rmi lstm-anomaly:latest

# Complete cleanup
docker system prune -a --volumes
```

## Next Steps

1. ✅ Start services: `docker-compose up -d`
2. ✅ Check logs: `docker-compose logs -f`
3. ✅ Verify metrics: `curl http://localhost:8428/api/v1/query?query=up`
4. ✅ Wait for anomaly detection: Check logs for "Training completed"
5. ✅ Query anomalies: `curl http://localhost:8428/api/v1/query?query=lstm_anomaly_score`

For more details, see:
- VictoriaMetrics: https://docs.victoriametrics.com/
- Docker Compose: https://docs.docker.com/compose/
