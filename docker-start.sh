#!/bin/bash
# Quick Start Script for LSTM Anomaly Detection with Docker

set -e

echo "🚀 LSTM Anomaly Detection - Docker Quick Start"
echo "================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null && ! docker-compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Use docker compose or docker-compose
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "✅ Docker is installed"
echo ""

# Step 1: Check required files
echo "📋 Step 1: Checking required files..."
required_files=("Dockerfile" "docker-compose.yml" "config.yaml" "requirements.txt" "prometheus.yml")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing file: $file"
        exit 1
    fi
    echo "   ✓ $file"
done
echo ""

# Step 2: Update config.yaml for Docker
echo "📝 Step 2: Checking config.yaml..."
if grep -q "localhost" config.yaml; then
    echo "⚠️  WARNING: config.yaml contains 'localhost'"
    echo "   For Docker, you should use 'victoriametrics' instead of 'localhost'"
    echo "   Example: datasource_url: \"http://victoriametrics:8428\""
    echo ""
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "   ✓ Config looks good"
fi
echo ""

# Step 3: Create necessary directories
echo "📁 Step 3: Creating directories..."
mkdir -p model_checkpoints
echo "   ✓ model_checkpoints/"
echo ""

# Step 4: Build Docker images
echo "🔨 Step 4: Building Docker images..."
$DOCKER_COMPOSE build
echo ""

# Step 5: Start services
echo "🎬 Step 5: Starting all services..."
$DOCKER_COMPOSE up -d
echo ""

# Step 6: Wait for services to be ready
echo "⏳ Step 6: Waiting for services to start..."
sleep 5

# Check if containers are running
echo ""
echo "📊 Container Status:"
$DOCKER_COMPOSE ps
echo ""

# Step 7: Verify VictoriaMetrics
echo "🔍 Step 7: Verifying VictoriaMetrics..."
max_retries=10
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -s http://localhost:8428/health > /dev/null 2>&1; then
        echo "   ✓ VictoriaMetrics is ready!"
        break
    fi
    retry_count=$((retry_count + 1))
    echo "   Waiting... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "   ⚠️  VictoriaMetrics might not be ready yet. Check logs: docker-compose logs victoriametrics"
fi
echo ""

# Step 8: Show useful information
echo "✨ Setup Complete!"
echo "=================="
echo ""
echo "📌 Access Points:"
echo "   • VictoriaMetrics UI:  http://localhost:8428"
echo "   • Node Exporter:       http://localhost:9100/metrics"
echo "   • VMAgent:             http://localhost:8429"
echo ""
echo "📜 Useful Commands:"
echo "   • View all logs:       $DOCKER_COMPOSE logs -f"
echo "   • View LSTM logs:      $DOCKER_COMPOSE logs -f lstm-anomaly"
echo "   • Stop all services:   $DOCKER_COMPOSE down"
echo "   • Restart LSTM:        $DOCKER_COMPOSE restart lstm-anomaly"
echo ""
echo "🔍 Test Queries:"
echo "   • Check metrics:       curl 'http://localhost:8428/api/v1/query?query=up'"
echo "   • Check anomalies:     curl 'http://localhost:8428/api/v1/query?query=lstm_anomaly_score'"
echo ""
echo "📖 For more details, see DOCKER_GUIDE.md"
echo ""

# Option to view logs
read -p "Would you like to view the logs now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Press Ctrl+C to exit logs"
    sleep 2
    $DOCKER_COMPOSE logs -f
fi
