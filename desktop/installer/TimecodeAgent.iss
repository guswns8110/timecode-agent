#define MyAppName "Timecode Agent"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Timecode Agent Desktop"

[Setup]
AppId={{D9D5B349-B0BD-420D-817F-1B8010CECEB8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Timecode Agent
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\build
OutputBaseFilename=TimecodeAgent-Setup-x64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\..\LICENSE
UninstallDisplayIcon={app}\runtime\Scripts\pythonw.exe
SetupLogging=yes
CloseApplications=yes

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Files]
Source: "..\tools\uv.exe"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "..\bootstrap\install-runtime.ps1"; DestDir: "{app}\bootstrap"; Flags: ignoreversion
Source: "..\bootstrap\launcher.pyw"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\pyproject.toml"; DestDir: "{app}\source"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}\source"; Flags: ignoreversion
Source: "..\..\README.ko.md"; DestDir: "{app}\source"; Flags: ignoreversion
Source: "..\..\src\*"; DestDir: "{app}\source\src"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\pyproject.toml"; DestDir: "{app}\source\desktop"; Flags: ignoreversion
Source: "..\timecode_desktop\*"; DestDir: "{app}\source\desktop\timecode_desktop"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\runtime\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\runtime\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "바탕화면 바로가기 만들기"; GroupDescription: "바로가기:"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\bootstrap\install-runtime.ps1"" -InstallRoot ""{app}"""; StatusMsg: "AI 런타임과 모델을 설치하고 있습니다. 네트워크 속도에 따라 시간이 걸릴 수 있습니다."; Flags: runhidden waituntilterminated
Filename: "{app}\runtime\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; Description: "{#MyAppName} 실행"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\source"
Type: filesandordirs; Name: "{app}\tools"
Type: filesandordirs; Name: "{app}\bootstrap"
Type: filesandordirs; Name: "{app}\ffmpeg"

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  FreeMB: Cardinal;
  TotalMB: Cardinal;
begin
  Result := '';
  if not GetSpaceOnDisk(ExpandConstant('{localappdata}'), True, FreeMB, TotalMB) then
    exit;
  if FreeMB < 15360 then
    Result := '설치하려면 최소 15GB의 여유 공간이 필요합니다.';
end;
