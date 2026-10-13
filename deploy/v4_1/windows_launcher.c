#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <stdio.h>
#include <wchar.h>

static int show_error(const wchar_t *message) {
    MessageBoxW(NULL, message, L"薇甘菊识别系统 V4.1", MB_OK | MB_ICONERROR);
    return 1;
}

static int runtime_ready(void) {
    const wchar_t *libraries[] = {L"msvcp140.dll", L"msvcp140_1.dll"};
    for (int i = 0; i < 2; i++) {
        HMODULE library = LoadLibraryExW(libraries[i], NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!library) return 0;
        FreeLibrary(library);
    }
    return 1;
}

int wmain(int argc, wchar_t **argv) {
    wchar_t root[32768], python[32768], command[32768], installer[32768];
    DWORD length = GetModuleFileNameW(NULL, root, 32768);
    if (length == 0 || length >= 32768) return show_error(L"无法读取程序路径。请完整解压到普通文件夹。");
    wchar_t *separator = wcsrchr(root, L'\\');
    if (!separator) return show_error(L"程序路径无效。");
    *separator = L'\0';
    if (wcslen(root) > 12000) return show_error(L"文件夹路径太长，请移到较短路径。");
    int self_test = 0, no_browser = 0;
    for (int i = 1; i < argc; i++) {
        if (!wcscmp(argv[i], L"--self-test")) self_test = 1;
        else if (!wcscmp(argv[i], L"--no-browser")) no_browser = 1;
        else return show_error(L"不支持的启动参数。");
    }
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    SetConsoleTitleW(L"薇甘菊 V4.1 — 请保持此窗口开启，关闭即停止识别");
    if (!runtime_ready()) {
        int answer = MessageBoxW(NULL,
            L"首次运行需要微软 Visual C++ 运行组件。\n完整包已附带微软官方安装器，无需联网。\n是否现在安装？可能需要管理员确认。",
            L"薇甘菊识别系统 · 首次运行", MB_YESNO | MB_ICONINFORMATION);
        if (answer != IDYES) return 1;
        swprintf(installer, 32768, L"%ls\\runtime\\VC_redist.x64.exe", root);
        SHELLEXECUTEINFOW info = {0};
        info.cbSize = sizeof(info); info.fMask = SEE_MASK_NOCLOSEPROCESS;
        info.lpVerb = L"runas"; info.lpFile = installer;
        info.lpParameters = L"/install /passive /norestart"; info.nShow = SW_SHOWNORMAL;
        if (!ShellExecuteExW(&info)) return show_error(L"运行组件安装未完成。请联系电脑管理员协助安装。无需安装Python。");
        WaitForSingleObject(info.hProcess, INFINITE);
        DWORD code = 1; GetExitCodeProcess(info.hProcess, &code); CloseHandle(info.hProcess);
        if ((code != 0 && code != 3010 && code != 1638) || !runtime_ready())
            return show_error(L"运行组件尚未就绪。如安装器提示重启，请重启后再次打开。");
    }
    swprintf(python, 32768, L"%ls\\runtime\\python\\python.exe", root);
    if (GetFileAttributesW(python) == INVALID_FILE_ATTRIBUTES)
        return show_error(L"运行文件不完整。请完整解压整个ZIP，不要只复制EXE，也不要在压缩包内直接运行。");
    swprintf(command, 32768, L"\"%ls\" -I \"%ls\\runtime\\launcher.py\"%ls%ls",
        python, root, self_test ? L" --self-test" : L"", no_browser ? L" --no-browser" : L"");
    HANDLE job = CreateJobObjectW(NULL, NULL);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {0};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
        return show_error(L"无法建立本机识别进程，请重试或联系管理员。");
    STARTUPINFOW start = {0}; start.cb = sizeof(start);
    PROCESS_INFORMATION process = {0};
    if (!CreateProcessW(python, command, NULL, NULL, FALSE, CREATE_SUSPENDED, NULL, root, &start, &process)) {
        CloseHandle(job);
        return show_error(L"无法启动内置运行环境。请完整解压程序，检查安全软件提示，并联系维护人员。");
    }
    if (!AssignProcessToJobObject(job, process.hProcess)) {
        TerminateProcess(process.hProcess, 1); CloseHandle(process.hThread);
        CloseHandle(process.hProcess); CloseHandle(job);
        return show_error(L"无法管理识别子进程，程序已安全停止。");
    }
    ResumeThread(process.hThread); CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1; GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hProcess); CloseHandle(job);
    if (exit_code && exit_code != 0xC000013A)
        show_error(L"识别系统启动失败，错误已保留。\n请打开 %LOCALAPPDATA%\\Mikania\\4.1\\logs，将日志发给维护人员。\n请勿关闭杀毒软件或系统安全保护。");
    return (int)exit_code;
}
