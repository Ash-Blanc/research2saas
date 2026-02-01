#!/usr/bin/env bash
# Quick start script for Research-to-SaaS Platform

echo "🚀 Starting Research-to-SaaS Backend..."
echo ""

# Kill any existing server on port 7777
echo "🔍 Checking for existing servers on port 7777..."
pkill -f "server.py" 2>/dev/null || true
sleep 1

echo "Backend will run on: http://localhost:7777"
echo ""
echo "Next steps:"
echo "1. Go to: https://os.agno.com"
echo "2. Sign in and click 'Add new OS'"
echo "3. Select 'Local' and enter: http://localhost:7777"
echo "4. Name it 'Research-to-SaaS' and click 'Connect'"
echo ""
echo "Starting server..."
echo ""

uv run python server.py
