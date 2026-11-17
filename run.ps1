<#
.SYNOPSIS
    ResearchIQ Windows Automation Script
.DESCRIPTION
    Helper script for installing dependencies, running unit tests, ingesting papers, or launching Streamlit.
.EXAMPLE
    .\run.ps1 -Action test
    .\run.ps1 -Action ingest -Target "research-data/2604.01733v1.pdf"
    .\run.ps1 -Action ingest -Target "all"
    .\run.ps1 -Action ui
#>

param(
    [ValidateSet("install", "test", "ingest", "ask", "ui", "eval", "security-demo", "clean")]
    [string]$Action = "ui",

    [string]$Target = "all",

    [string]$Question = "",

    [switch]$Debug = $false
)

$ErrorActionPreference = "Stop"

# Ensure UTF-8 output in PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

switch ($Action) {
    "install" {
        Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
        python -m pip install -r requirements.txt
    }
    "test" {
        Write-Host "Running ResearchIQ Test Suite..." -ForegroundColor Cyan
        python tests/run_tests.py
    }
    "ingest" {
        if ($Target -eq "all") {
            Write-Host "Ingesting all research papers from research-data/..." -ForegroundColor Cyan
            python src/ingest_cli.py --all
        } else {
            Write-Host "Ingesting single paper: $Target..." -ForegroundColor Cyan
            python src/ingest_cli.py --paper "$Target"
        }
    }
    "ask" {
        if (-not $Question) {
            Write-Host "Please provide a question via -Question '...'" -ForegroundColor Red
            return
        }
        $debugFlag = if ($Debug) { "--debug" } else { "" }
        python src/ask_cli.py --question "$Question" $debugFlag
    }
    "ui" {
        Write-Host "Launching ResearchIQ Streamlit Web Interface..." -ForegroundColor Cyan
        python -m streamlit run src/ui/app.py
    }
    "eval" {
        Write-Host "Running ResearchIQ Mini-Evaluation Suite..." -ForegroundColor Cyan
        python -m src.eval.run_eval
    }
    "security-demo" {
        Write-Host "Running Prompt Injection Security Demonstration..." -ForegroundColor Cyan
        python scripts/injection_demo.py
    }
    "clean" {
        Write-Host "Cleaning index and figure cache directories..." -ForegroundColor Cyan
        if (Test-Path "data\index") {
            Remove-Item -Path "data\index\*" -Recurse -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path "data\figures") {
            Remove-Item -Path "data\figures\*" -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Host "Cache cleaned." -ForegroundColor Green
    }
}
