# Push changes (run outside sandbox via: powershell -ExecutionPolicy Bypass -File "d:\USERFILES\GitHub\sd-webui-forge-classic-neo\do_push.ps1")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".git\index.lock") { Remove-Item ".git\index.lock" -Force }

git add backend/nn/_qwen_lora.py
git status
git commit -m "LoRA: PEFT format detection and skip log (v2.3.8)"
git push
