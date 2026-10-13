/* Self-extracting, per-user installer. ZIP payload and footer appended by builder. */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>

static int error(const wchar_t *message) {
    MessageBoxW(NULL, message, L"薇甘菊 V4.1 安装", MB_OK | MB_ICONERROR);
    return 1;
}

int wmain(void) {
    wchar_t self[32768], tempdir[MAX_PATH], temp[MAX_PATH], archive[MAX_PATH + 5];
    wchar_t system[MAX_PATH], command[16000], installed[32768];
    if (MessageBoxW(NULL,
        L"将安装薇甘菊识别系统 V4.1 到当前用户的程序目录，并创建桌面快捷方式。\n"
        L"已包含模型和Python环境，无需联网下载。适用于Windows 10/11 64位。\n"
        L"请预留至少1GB空间。本版本尚待Windows实机验收，供测试使用。\n\n是否开始？",
        L"薇甘菊 V4.1 离线安装", MB_YESNO | MB_ICONQUESTION) != IDYES) return 0;
    DWORD length = GetModuleFileNameW(NULL, self, 32768);
    if (!length || length >= 32768) return error(L"无法读取安装包位置。");
    FILE *source = _wfopen(self, L"rb");
    if (!source) return error(L"无法打开安装包。");
    unsigned char magic[8]; uint64_t size = 0;
    if (_fseeki64(source, -16, SEEK_END) || fread(magic, 1, 8, source) != 8 ||
        memcmp(magic, "MIKANIA1", 8) || fread(&size, 8, 1, source) != 1 ||
        size < 1000000 || size > 2000000000ULL || _fseeki64(source, -(int64_t)(size + 16), SEEK_END)) {
        fclose(source); return error(L"安装包不完整，请重新复制完整文件。");
    }
    if (!GetTempPathW(MAX_PATH, tempdir) || !GetTempFileNameW(tempdir, L"MKN", 0, temp)) {
        fclose(source); return error(L"无法建立临时文件，请检查磁盘空间。");
    }
    swprintf(archive, MAX_PATH + 5, L"%ls.zip", temp);
    if (!MoveFileW(temp, archive)) { fclose(source); DeleteFileW(temp); return error(L"临时文件创建失败。"); }
    FILE *output = _wfopen(archive, L"wb");
    if (!output) { fclose(source); DeleteFileW(archive); return error(L"无法写入临时文件。"); }
    unsigned char buffer[65536]; uint64_t remaining = size; int ok = 1;
    while (remaining) {
        size_t amount = remaining > sizeof(buffer) ? sizeof(buffer) : (size_t)remaining;
        if (fread(buffer, 1, amount, source) != amount || fwrite(buffer, 1, amount, output) != amount) { ok = 0; break; }
        remaining -= amount;
    }
    fclose(source); if (fclose(output)) ok = 0;
    if (!ok) { DeleteFileW(archive); return error(L"安装包复制失败，请检查磁盘空间。"); }
    SetEnvironmentVariableW(L"MIKANIA_INSTALL_ARCHIVE", archive);
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleTitleW(L"薇甘菊 V4.1 正在安装，请稍候");
    /* Paths pass via environment or OS APIs, never interpolate user text into PowerShell code. */
    GetSystemDirectoryW(system, MAX_PATH);
    swprintf(command, 16000,
        L"\"%ls\\WindowsPowerShell\\v1.0\\powershell.exe\" -NoLogo -NoProfile -NonInteractive -Command \""
        L"$ErrorActionPreference='Stop'; try { "
        L"$target=Join-Path $env:LOCALAPPDATA 'Programs\\Mikania\\4.1.0-20261013'; "
        L"$log=Join-Path $env:LOCALAPPDATA 'Mikania\\4.1\\logs'; New-Item -ItemType Directory -Force -Path $log | Out-Null; "
        L"if(Test-Path -LiteralPath $target){throw 'Installation directory already exists. Use the existing shortcut, or rename the previous installation before retrying.'}; "
        L"New-Item -ItemType Directory -Path $target | Out-Null; "
        L"Expand-Archive -LiteralPath $env:MIKANIA_INSTALL_ARCHIVE -DestinationPath $target; "
        L"$root=Join-Path $target 'mikania-windows-v4.1.0'; "
        L"$exe=Join-Path $root 'Mikania.exe'; if(!(Test-Path -LiteralPath $exe)){throw 'Incomplete package'}; "
        L"$ws=New-Object -ComObject WScript.Shell; "
        L"$link=$ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Mikania V4.1.lnk')); "
        L"$link.TargetPath=$exe; $link.WorkingDirectory=$root; $link.Description='Mikania V4.1 Offline'; $link.Save(); "
        L"'Installation completed' | Out-File -Encoding utf8 (Join-Path $log 'install.log'); exit 0 "
        L"} catch { $_ | Out-File -Encoding utf8 (Join-Path $env:TEMP 'mikania-install-error.txt'); Write-Error $_; exit 1 }\"", system);
    STARTUPINFOW start = {0}; start.cb = sizeof(start); PROCESS_INFORMATION process = {0};
    if (!CreateProcessW(NULL, command, NULL, NULL, FALSE, 0, NULL, NULL, &start, &process)) {
        DeleteFileW(archive); return error(L"无法启动Windows自带解压程序。可改用同版本ZIP手动解压。");
    }
    CloseHandle(process.hThread); WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1; GetExitCodeProcess(process.hProcess, &code); CloseHandle(process.hProcess);
    DeleteFileW(archive);
    if (code) return error(L"安装未完成。请查看 %TEMP%\\mikania-install-error.txt。\n"
        L"如已安装过，可直接使用桌面快捷方式；重复安装不会覆盖旧文件。\n也可使用ZIP版完整解压后运行。");
    if (MessageBoxW(NULL, L"安装完成，桌面已生成 Mikania V4.1 快捷方式。\n是否现在启动？",
        L"薇甘菊 V4.1", MB_YESNO | MB_ICONINFORMATION) == IDYES) {
        DWORD count = GetEnvironmentVariableW(L"LOCALAPPDATA", installed, 25000);
        if (!count || count >= 25000) return error(L"无法读取用户目录，请从桌面快捷方式启动。");
        wcscat(installed, L"\\Programs\\Mikania\\4.1.0-20261013\\mikania-windows-v4.1.0\\Mikania.exe");
        if ((INT_PTR)ShellExecuteW(NULL, L"open", installed, NULL, NULL, SW_SHOWNORMAL) <= 32)
            return error(L"自动启动未成功，请双击桌面 Mikania V4.1 快捷方式。");
    }
    return 0;
}
