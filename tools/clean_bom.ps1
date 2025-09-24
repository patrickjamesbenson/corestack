# tools\clean_bom.ps1
# Removes BOM/zero-width/bidi chars and normalises line endings.
param(
  [string[]]$Exts = @("*.py","*.json","*.txt","*.csv","*.cfg","*.ini")
)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$bad  = @("`uFEFF","`u200B","`u200C","`u200D","`u200E","`u200F","`u202A","`u202B","`u202C","`u202D","`u202E")
$changed = @()

Get-ChildItem -Path $root -Recurse -File -Include $Exts | ForEach-Object {
  try { $s = Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 }
  catch { return }
  $orig = $s
  foreach($ch in $bad){ $s = $s -replace $ch, "" }
  # normalise line endings
  $s = $s -replace "`r`n","`n" -replace "`r","`n"
  if($s -ne $orig){
    Set-Content -LiteralPath $_.FullName -Value $s -Encoding UTF8
    $changed += $_.FullName
  }
}

"Cleaned $($changed.Count) files."
$changed | ForEach-Object { "  $_" }
