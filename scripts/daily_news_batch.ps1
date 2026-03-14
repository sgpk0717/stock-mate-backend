# StockMate 일일 뉴스 수집 배치 스크립트
# Windows Task Scheduler에서 18:00에 호출됨
#
# 장 마감 후 충분히 시간을 두고 뉴스를 수집하여
# 15:30~18:00 사이 발행된 기사도 포함한다.

$ErrorActionPreference = "Continue"
$ProjectDir = "C:\Users\Rex\stock-mate\stock-mate-backend"
$LogDir = "$ProjectDir\logs\catchup"
$Date = Get-Date -Format "yyyy-MM-dd"
$LogFile = "$LogDir\$Date-news.log"
$ContainerName = "stockmate-catchup-news"

# 로그 디렉토리 생성
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Log {
    param([string]$Message)
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Log "=== StockMate 뉴스 수집 시작 ==="

# 1) Docker 실행 확인
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Docker not running" }
} catch {
    Log "ERROR: Docker가 실행되지 않음 — 종료"
    exit 1
}

# 2) DB 헬스체크
$dbHealth = docker inspect --format "{{.State.Health.Status}}" stockmate-db 2>&1
if ($dbHealth -ne "healthy") {
    Log "ERROR: DB 헬스체크 실패 ($dbHealth) — 종료"
    exit 1
}

# 3) 중복 확인
$running = docker ps --filter "name=$ContainerName" --format "{{.ID}}" 2>&1
if ($running -and $running -ne "") {
    Log "이미 실행 중 ($ContainerName) — 종료"
    exit 0
}

# 4) 뉴스 수집 실행
Log "catchup_external.py --jobs news 실행 시작..."
Set-Location $ProjectDir

$output = docker compose run --rm --name $ContainerName app python -m scripts.catchup_external --jobs news 2>&1
$exitCode = $LASTEXITCODE

# 로그 기록 (줄당 500자 트룬케이트)
if ($output) {
    foreach ($line in $output) {
        $s = "$line"
        if ($s -match '^\s*(Container|Network)\s' -or $s -match '^time=' -or $s -match '^\s*$') { continue }
        if ($s.Length -gt 500) { $s = $s.Substring(0, 500) + "...[truncated]" }
        Add-Content -Path $LogFile -Value $s
    }
}

if ($exitCode -eq 0) {
    Log "=== 뉴스 수집 성공 (exit=$exitCode) ==="
} else {
    Log "=== 뉴스 수집 실패 (exit=$exitCode) ==="
}

exit $exitCode
