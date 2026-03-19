#!/bin/bash
# Start TotalSegmentator on Apple Silicon Mac (M1/M2/M3)

echo "🍎 Starting TotalSegmentator on Apple Silicon..."
echo ""

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: This script is optimized for Apple Silicon (ARM64)"
    echo "   Your architecture: $(uname -m)"
    echo ""
fi

# Clean up old containers
echo "🧹 Cleaning up old containers..."
docker compose down

# Build and start services
echo ""
echo "🏗️  Building Apple Silicon optimized images..."
docker compose build --no-cache

echo ""
echo "🚀 Starting services..."
docker compose up

# Note: Press Ctrl+C to stop
