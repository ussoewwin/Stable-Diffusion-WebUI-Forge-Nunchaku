$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:http_proxy = ""
$env:https_proxy = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
Set-Location $PSScriptRoot
gh release edit 1.4.1 --title "1.4.1 - transformers 5+ Compatibility Patch"
