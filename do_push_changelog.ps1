$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
git add Changelog/CHANGELOG.md
git commit -m "Changelog: add Version 1.4.0"
git push
