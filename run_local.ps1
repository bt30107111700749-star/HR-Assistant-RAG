$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path "venv\Scripts\python.exe")) {
    py -3.11 -m venv venv
}

& "venv\Scripts\python.exe" -m pip install --upgrade pip
& "venv\Scripts\python.exe" -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env. Add your OPENROUTER_API_KEY before asking questions." -ForegroundColor Yellow
}

& "venv\Scripts\python.exe" -m streamlit run streamlit_app.py
