# tools\scan_bom.ps1
# Lists files with BOM/zero-width/bidi chars.
param(
  [string[]]$Exts = @("*.py","*.json","*.txt","*.csv","*.cfg","*.ini")
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Write-Host "Scanning under $root`n"

# A) Files that START with UTF-8 BOM EF BB BF
$startBOM = Get-ChildItem -Path $root -Recurse -File -Include $Exts |
  Where-Object {
    $b = [IO.File]::ReadAllBytes($_.FullName)
    ($b.Length -ge 3 -and $b[0]-eq 239 -and $b[1]-eq 187 -and $b[2]-eq 191)
  } | Select-Object -ExpandProperty FullName

Write-Host "-- Files with START-OF-FILE BOM --"
$startBOM | ForEach-Object { "  $_" }

# B) Files containing any invisible chars anywhere
$pattern = '\uFEFF|\u200B|\u200C|\u200D|\u200E|\u200F|\u202A|\u202B|\u202C|\u202D|\u202E'
$midBOM = Get-ChildItem -Path $root -Recurse -File -Include $Exts |
  Where-Object {
    try { $s = Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 } catch { $s = '' }
    $s -match $pattern
  } | Select-Object -ExpandProperty FullName -Unique

Write-Host "`n-- Files with EMBEDDED BOM/zero-width/bidi marks --"
$midBOM | ForEach-Object { "  $_" }

Write-Host "`nScans complete."
