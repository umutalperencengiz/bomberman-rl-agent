<#
.SYNOPSIS
  Ajani GUI'de izle. PowerShell tirnak/ortam-degiskeni derdini ortadan kaldirir.

.DESCRIPTION
  PowerShell 5.1'de `&&` yok, `VAR=x komut` yok, ve tirnakla baslayan bir satir
  komut degil "string ifadesi" sayilir (`&` call operator'u gerekir). Bu script
  hepsini kapsuller.

.EXAMPLE
  .\watch.ps1
  # varsayilan config (agent_code/irmak_umut/config.yaml) ile classic

.EXAMPLE
  .\watch.ps1 -Config configs/_seeds/m02a_tab_direct_s0.yaml -TurnBased
  # Task 2 modelini adim adim izle (her tusa basista bir adim)

.EXAMPLE
  .\watch.ps1 -Agents rule_based_agent -Interval 0.35
  # referans ajani izle

.EXAMPLE
  .\watch.ps1 -Config configs/e07b_living.yaml -Scenario coin-heaven
  # Task 1 kazananini izle
#>
param(
    [string]   $Config    = "",
    [string]   $Scenario  = "classic",
    [string[]] $Agents    = @("irmak_umut"),
    [double]   $Interval  = 0.35,
    [int]      $Rounds    = 10,
    [switch]   $TurnBased,
    [switch]   $NoGui,
    [string]   $Python    = "C:\Users\Irmak\miniconda3\envs\ml_homework\python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path $Python)) {
    Write-Error "Python bulunamadi: $Python  (-Python ile yol verebilirsin)"
}

# BBRL_CONFIG'i YALNIZCA bu cagri icin ayarla, sonra eski haline dondur.
# Aksi halde oturumda asili kalir ve sonraki komutlar yanlis modeli yukler.
$prev = $env:BBRL_CONFIG
try {
    if ($Config -ne "") {
        if (-not (Test-Path $Config)) { Write-Error "Config yok: $Config" }
        $env:BBRL_CONFIG = $Config
        Write-Host "config : $Config" -ForegroundColor Cyan
    } else {
        Remove-Item Env:BBRL_CONFIG -ErrorAction SilentlyContinue
        Write-Host "config : agent_code/<ajan>/config.yaml (varsayilan)" -ForegroundColor Cyan
    }

    $cmdArgs = @("main.py", "play", "--agents") + $Agents +
               @("--scenario", $Scenario, "--n-rounds", "$Rounds")
    if ($NoGui)     { $cmdArgs += "--no-gui" }
    else            { $cmdArgs += @("--update-interval", "$Interval") }
    if ($TurnBased) { $cmdArgs += "--turn-based" }

    Write-Host "ajanlar: $($Agents -join ', ')" -ForegroundColor Cyan
    Write-Host "senaryo: $Scenario" -ForegroundColor Cyan
    if ($TurnBased) {
        Write-Host "ADIM ADIM: her tusa basista bir adim ilerler" -ForegroundColor Yellow
    }
    Write-Host ""

    & $Python @cmdArgs
}
finally {
    if ($null -eq $prev) { Remove-Item Env:BBRL_CONFIG -ErrorAction SilentlyContinue }
    else                 { $env:BBRL_CONFIG = $prev }
}
