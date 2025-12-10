# PowerShell script to start Langflow backend
# This script properly starts the Langflow backend on Windows with PowerShell

Write-Host "Starting Langflow backend..." -ForegroundColor Green

# Change to the project directory
Set-Location -Path "d:\langflow_spark\langflow-AI"

# Check if .env file exists, if not create from example
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

# Start the backend using uv
Write-Host "Launching Uvicorn server..." -ForegroundColor Green
uv run uvicorn --factory langflow.main:create_app --host 0.0.0.0 --port 7860 --reload --env-file .env --loop asyncio
