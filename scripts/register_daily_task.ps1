# StockMate 일일 캐치업 — Windows Task Scheduler 등록
#
# 관리자 권한으로 실행:
#   powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1
#
# 등록 후 확인:
#   schtasks /query /tn StockMate-DailyCatchup /v
#
# 삭제:
#   schtasks /delete /tn StockMate-DailyCatchup /f

$ErrorActionPreference = "Stop"

$TaskName = "StockMate-DailyCatchup"
$ProjectDir = "C:\Users\Rex\stock-mate\stock-mate-backend"
$ScriptPath = "$ProjectDir\scripts\daily_batch.ps1"

# 관리자 권한 확인
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: 관리자 권한이 필요합니다." -ForegroundColor Red
    Write-Host "  우클릭 → '관리자 권한으로 실행'으로 다시 실행하세요."
    exit 1
}

# 기존 태스크 삭제 (있으면)
$existing = schtasks /query /tn $TaskName 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "기존 태스크 '$TaskName' 삭제 중..."
    schtasks /delete /tn $TaskName /f | Out-Null
}

# 태스크 등록
Write-Host "태스크 '$TaskName' 등록 중..."

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`"" `
    -WorkingDirectory $ProjectDir

$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At "16:45"

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10)

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "StockMate 일일 캔들 데이터 캐치업 (장 마감 후 16:45 실행, 놓친 스케줄 부팅 시 즉시 실행)" `
    | Out-Null

# 확인
Write-Host ""
Write-Host "=== 캔들+수급 태스크 등록 완료 ===" -ForegroundColor Green
Write-Host ""
Write-Host "태스크: $TaskName"
Write-Host "실행 시각: 매일 16:45"
Write-Host "스크립트: $ScriptPath"
Write-Host ""

# ── 뉴스 수집 태스크 (18:00) ──

$NewsTaskName = "StockMate-DailyNews"
$NewsScriptPath = "$ProjectDir\scripts\daily_news_batch.ps1"

# 기존 뉴스 태스크 삭제 (있으면)
$existingNews = schtasks /query /tn $NewsTaskName 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "기존 태스크 '$NewsTaskName' 삭제 중..."
    schtasks /delete /tn $NewsTaskName /f | Out-Null
}

Write-Host "태스크 '$NewsTaskName' 등록 중..."

$NewsAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$NewsScriptPath`"" `
    -WorkingDirectory $ProjectDir

$NewsTrigger = New-ScheduledTaskTrigger `
    -Daily `
    -At "18:00"

$NewsSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5)

Register-ScheduledTask `
    -TaskName $NewsTaskName `
    -Action $NewsAction `
    -Trigger $NewsTrigger `
    -Settings $NewsSettings `
    -Principal $Principal `
    -Description "StockMate 일일 뉴스 수집 + 감성 분석 (18:00 실행, 장 마감 후 뉴스 포함)" `
    | Out-Null

Write-Host ""
Write-Host "=== 뉴스 태스크 등록 완료 ===" -ForegroundColor Green
Write-Host ""
Write-Host "태스크: $NewsTaskName"
Write-Host "실행 시각: 매일 18:00"
Write-Host "스크립트: $NewsScriptPath"
Write-Host ""

Write-Host "주요 설정 (공통):"
Write-Host "  - StartWhenAvailable = true  (놓친 스케줄 부팅 즉시 실행)"
Write-Host "  - MultipleInstances = IgnoreNew  (중복 실행 방지)"
Write-Host "  - RestartOnFailure = 2회"
Write-Host ""
Write-Host "확인: schtasks /query /tn $TaskName /v"
Write-Host "      schtasks /query /tn $NewsTaskName /v"
Write-Host "삭제: schtasks /delete /tn $TaskName /f"
Write-Host "      schtasks /delete /tn $NewsTaskName /f"
Write-Host "수동: schtasks /run /tn $TaskName"
Write-Host "      schtasks /run /tn $NewsTaskName"
