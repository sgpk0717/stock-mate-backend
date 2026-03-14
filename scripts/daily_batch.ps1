# StockMate 일일 캔들 캐치업 배치 스크립트
# Windows Task Scheduler에서 호출됨
#
# 기능:
#   1. Docker 실행 확인 (대기 최대 5분)
#   2. 중복 컨테이너 확인 → 이미 실행 중이면 종료
#   3. catchup_candles.py 실행
#   4. 로그 파일 저장

$ErrorActionPreference = "Continue"
$ProjectDir = "C:\Users\Rex\stock-mate\stock-mate-backend"
$LogDir = "$ProjectDir\logs\catchup"
$Date = Get-Date -Format "yyyy-MM-dd"
$LogFile = "$LogDir\$Date.log"
$ContainerName = "stockmate-catchup"

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

Log "=== StockMate 일일 캐치업 시작 ==="

# 1) Docker 실행 확인
$maxWait = 300  # 5분
$waited = 0
while ($true) {
    try {
        $null = docker info 2>&1
        if ($LASTEXITCODE -eq 0) {
            break
        }
    } catch {}

    if ($waited -ge $maxWait) {
        Log "ERROR: Docker가 $maxWait 초 내에 시작되지 않음 — 종료"
        exit 1
    }

    if ($waited -eq 0) {
        Log "Docker 대기 중..."
        # Docker Desktop 시작 시도
        $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dockerDesktop) {
            Start-Process $dockerDesktop
        }
    }
    Start-Sleep -Seconds 10
    $waited += 10
}

Log "Docker 실행 확인 OK"

# 2) 중복 컨테이너 확인
$running = docker ps --filter "name=$ContainerName" --format "{{.ID}}" 2>&1
if ($running -and $running -ne "") {
    Log "이미 실행 중 ($ContainerName) — 종료"
    exit 0
}

# 3) DB 컨테이너 헬스체크 대기
$dbWait = 0
while ($true) {
    $dbHealth = docker inspect --format "{{.State.Health.Status}}" stockmate-db 2>&1
    if ($dbHealth -eq "healthy") {
        break
    }
    if ($dbWait -ge 120) {
        Log "ERROR: DB 헬스체크 실패 — 종료"
        exit 1
    }
    if ($dbWait -eq 0) {
        Log "DB 헬스체크 대기 중..."
        # Docker Compose 서비스 시작
        Set-Location $ProjectDir
        docker compose up -d postgres 2>&1 | Out-Null
    }
    Start-Sleep -Seconds 5
    $dbWait += 5
}

# 4) catchup 실행
Log "catchup_candles.py 실행 시작..."
Set-Location $ProjectDir

# ErrorActionPreference=Continue 이므로 docker stderr 경고가 스크립트를 중단하지 않음
$output = docker compose run --rm --name $ContainerName app python -m scripts.catchup_candles 2>&1
$exitCode = $LASTEXITCODE

# 로그에 출력 기록 (docker compose 내부 메시지 제외)
if ($output) {
    foreach ($line in $output) {
        $s = "$line"
        if ($s -match '^\s*(Container|Network)\s' -or $s -match '^time=' -or $s -match '^\s*$') { continue }
        Add-Content -Path $LogFile -Value $s
    }
}

if ($exitCode -eq 0) {
    Log "=== 캔들 캐치업 성공 (exit=$exitCode) ==="
} else {
    Log "=== 캔들 캐치업 실패 (exit=$exitCode) ==="
}

# 5) 외부 데이터 캐치업 (투자자 수급 / 공매도·신용 / 뉴스) — 별도 컨테이너
$ExtContainerName = "stockmate-catchup-ext"
$ExtLogFile = "$LogDir\$Date-external.log"

# 이미 실행 중인지 확인
$extRunning = docker ps --filter "name=$ExtContainerName" --format "{{.ID}}" 2>&1
if ($extRunning -and $extRunning -ne "") {
    Log "[외부 데이터] 이미 실행 중 ($ExtContainerName) — 스킵"
} else {
    Log "[외부 데이터] catchup_external.py 실행 시작..."

    $extOutput = docker compose run --rm --name $ExtContainerName app python -m scripts.catchup_external --jobs investor,margin_short 2>&1
    $extExitCode = $LASTEXITCODE

    # 로그 기록 (Docker 내부 메시지 제외, 줄당 500자 트룬케이트)
    if ($extOutput) {
        foreach ($line in $extOutput) {
            $s = "$line"
            if ($s -match '^\s*(Container|Network)\s' -or $s -match '^time=' -or $s -match '^\s*$') { continue }
            if ($s.Length -gt 500) { $s = $s.Substring(0, 500) + "...[truncated]" }
            Add-Content -Path $ExtLogFile -Value $s
        }
    }

    if ($extExitCode -eq 0) {
        Log "[외부 데이터] 성공 (exit=$extExitCode)"
    } else {
        Log "[외부 데이터] 실패 (exit=$extExitCode) — 상세: $ExtLogFile"
    }
}

exit $exitCode
