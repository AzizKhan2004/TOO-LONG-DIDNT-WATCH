# TL-DW GitHub Push Script
# Run this script after installing Git

Write-Host "=== TL-DW GitHub Push Script ===" -ForegroundColor Green
Write-Host ""

# Set hardcoded repo URL - MODIFY THIS if you want a different repo
$remoteUrl = "https://github.com/AzizKhan2004/TOO-LONG-DIDNT-WATCH.git"
$commitMessage = "Full project upload"

$gitVersion = git --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit
}

Write-Host "✓ Git is installed: $gitVersion" -ForegroundColor Green
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan

# Initialize Git repository
Write-Host "Step 1: Initializing Git repository..." -ForegroundColor Yellow
git init

# Add all files except those in .gitignore
Write-Host "Step 2: Adding all files to Git..." -ForegroundColor Yellow
git add .

# Commit
Write-Host "Step 3: Committing changes..." -ForegroundColor Yellow
git commit -m "$commitMessage"

# Set remote and push
Write-Host "Step 4: Setting remote and pushing to GitHub..." -ForegroundColor Yellow
git branch -M main
git remote remove origin 2>$null
git remote add origin $remoteUrl
git push -u origin main

Write-Host "\n✓ Push successful!" -ForegroundColor Green
Write-Host "\nScript complete."



