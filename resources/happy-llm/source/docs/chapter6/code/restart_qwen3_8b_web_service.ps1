$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..\..\..\..\..\..')).Path
$ServiceScript = Join-Path $ScriptDir 'qwen3_8b_web_service.py'
$OutLog = Join-Path $RepoRoot 'qwen3_web_service.out.log'
$ErrLog = Join-Path $RepoRoot 'qwen3_web_service.err.log'
$Port = 8000

$connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($connections) {
    $connections |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object {
            Stop-Process -Id $_ -Force
        }
}

Start-Sleep -Seconds 1

foreach ($log in @($OutLog, $ErrLog)) {
    if (Test-Path $log) {
        Remove-Item $log -Force
    }
}

Start-Process -FilePath python -ArgumentList @(
    $ServiceScript,
    '--host', '127.0.0.1',
    '--port', '8000'
) -WorkingDirectory $RepoRoot -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -WindowStyle Hidden

Start-Sleep -Seconds 5

Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, OwningProcess
