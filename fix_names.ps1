$ErrorActionPreference = "Stop"

# On prend seulement les dossiers utiles (src, flows, et racine)
$files = Get-ChildItem -Recurse -Include .py -Path .\src, .\flows, . | Where-Object { -not ($_.FullName -like "\.venv*") }

foreach ($f in $files) {
    $text = Get-Content -Path $f.FullName -Raw
    $orig = $text

    # Corrige la garde mal écrite
    $text = $text -replace 'if\s*name\s*==\s*"main"\s*:', 'if _name_ == "_main_":'

    # Corrige aussi les variantes (file, init)
    $text = $text -replace '\b_file_\b', '_file_'
    $text = $text -replace '\b_init_\b', '_init_'

    if ($text -ne $orig) {
        Set-Content -Path $f.FullName -Value $text -Encoding UTF8
        Write-Host "Fixed: $($f.FullName)"
    }
}

Write-Host "`n--- Lignes suspectes restantes ---"
$files | Select-String -Pattern "name|main" | Select-Object Path, LineNumber, Line
