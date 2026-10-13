$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundleRoot = Split-Path -Parent $Root
$ToolsDir = Join-Path $Root ".runtime-tools"
$PythonDir = Join-Path $Root ".runtime-python"
$RuntimeDir = Join-Path $Root ".runtime"
$LogsDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogsDir "install-windows.log"
$UvVersion = "0.11.32"
$UvExe = Join-Path $ToolsDir "uv.exe"
$PythonExe = Join-Path $RuntimeDir "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $ToolsDir, $PythonDir, $LogsDir | Out-Null
Start-Transcript -Path $LogFile -Append | Out-Null

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "命令执行失败（退出码 $LASTEXITCODE）：$Program"
    }
}

try {
    $RequiredFiles = @(
        (Join-Path $Root "launcher.py"),
        (Join-Path $Root "requirements.txt"),
        (Join-Path $BundleRoot "模型文件\mikania-yolo26s-v2\model.onnx"),
        (Join-Path $BundleRoot "程序源码\src\mikania_app\main.py")
    )
    foreach ($RequiredFile in $RequiredFiles) {
        if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
            throw "交付包文件不完整：$RequiredFile。请先对 ZIP 选择‘全部解压’，再运行安装。"
        }
    }

    Write-Host "[1/4] 准备环境安装工具……" -ForegroundColor Cyan
    if (-not (Test-Path $UvExe)) {
        $env:UV_UNMANAGED_INSTALL = $ToolsDir
        $env:UV_NO_MODIFY_PATH = "1"
        $installerUrl = "https://astral.sh/uv/$UvVersion/install.ps1"
        Invoke-RestMethod -Uri $installerUrl -UseBasicParsing | Invoke-Expression
    }
    if (-not (Test-Path $UvExe)) {
        throw "环境安装工具下载失败。"
    }

    $env:UV_PYTHON_INSTALL_DIR = $PythonDir
    $env:UV_LINK_MODE = "copy"
    $env:UV_NO_MODIFY_PATH = "1"

    Write-Host "[2/4] 自动下载独立 Python 运行时……" -ForegroundColor Cyan
    Invoke-Checked $UvExe "python" "install" "3.11"

    Write-Host "[3/4] 创建系统专用运行环境……" -ForegroundColor Cyan
    Invoke-Checked $UvExe "venv" $RuntimeDir "--python" "3.11" "--managed-python" "--clear"

    Write-Host "[4/4] 安装识别组件并检查模型……" -ForegroundColor Cyan
    Invoke-Checked $UvExe "pip" "install" "--python" $PythonExe "-r" (Join-Path $Root "requirements.txt")
    Invoke-Checked $PythonExe (Join-Path $Root "launcher.py") "--self-test"

    $installed = @(
        "Mikania portable runtime installed",
        "version=2.1.0",
        "python=3.11",
        "installed_at=$([DateTime]::Now.ToString('s'))"
    )
    Set-Content -Path (Join-Path $Root ".installed-ok") -Value $installed -Encoding UTF8
    Write-Host "环境安装完成。" -ForegroundColor Green
}
catch {
    Write-Host "安装失败：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "安装日志：$LogFile" -ForegroundColor Yellow
    exit 1
}
finally {
    try { Stop-Transcript | Out-Null } catch { }
}
