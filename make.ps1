# Windows'ta `make` yok. Ayni hedefleri calistiran ince kabuk.
#   .\make.ps1 test
#   .\make.ps1 lint
#   .\make.ps1 analyze -V data\raw\ornek.mp4 -E jump -L beginner
#   .\make.ps1 overlay -V data\raw\ornek.mp4
param(
    [Parameter(Position = 0)][string]$Target = "help",
    [string]$V = "",
    [string]$E = "jump",
    [string]$L = "beginner",
    [string]$D = "dataaw"
)

$ErrorActionPreference = "Stop"
$py = "python"

switch ($Target) {
    "install" { & $py -m pip install -e ".[core,dev]" }
    "test"    { & $py -m pytest }
    "lint"    { & $py -m ruff check .; if ($?) { & $py -m mypy } }
    "fmt"     { & $py -m ruff format .; & $py -m ruff check . --fix }
    "dev"     { & $py -m uvicorn api.main:app --reload }
    "analyze" { & $py -m core.cli analyze --video $V --element $E --level $L }
    "overlay" { & $py -m core.cli overlay --video $V }
    "check"   { if ($V) { & $py -m core.cli check --video $V } else { & $py -m core.cli check --dir $D } }
    default {
        Write-Output "hedefler: install | test | lint | fmt | dev | analyze | overlay | check"
    }
}
