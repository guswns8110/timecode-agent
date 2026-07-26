#define MyAppName "Timecode Agent"
#define MyAppVersion "0.1.6"
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
Source: "..\python-runtime\*"; DestDir: "{app}\python-runtime-v3"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\bootstrap\install-runtime.ps1"; DestDir: "{app}\bootstrap"; Flags: ignoreversion
Source: "..\bootstrap\launcher.pyw"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\pyproject.toml"; DestDir: "{app}\source"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}\source"; Flags: ignoreversion
Source: "..\..\README.ko.md"; DestDir: "{app}\source"; Flags: ignoreversion
Source: "..\..\docs\public\README.md"; DestDir: "{app}\source\docs\public"; Flags: ignoreversion
Source: "..\..\src\*"; DestDir: "{app}\source\src"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\pyproject.toml"; DestDir: "{app}\source\desktop"; Flags: ignoreversion
Source: "..\timecode_desktop\*"; DestDir: "{app}\source\desktop\timecode_desktop"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\runtime\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\runtime\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "바탕화면 바로가기 만들기"; GroupDescription: "바로가기:"

[Run]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\bootstrap\install-runtime.ps1"" -InstallRoot ""{app}"""; WorkingDir: "{app}"; StatusMsg: "AI 런타임과 모델을 설치하고 있습니다. 다운로드 창을 닫지 마세요."; Flags: waituntilterminated; AfterInstall: VerifyRuntime; Check: ShouldInstallRuntime
Filename: "{app}\runtime\Scripts\pythonw.exe"; Parameters: """{app}\launcher.pyw"""; WorkingDir: "{app}"; Description: "{#MyAppName} 실행"; Flags: nowait postinstall skipifsilent; Check: RuntimeReady

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\python-runtime-v3"
Type: filesandordirs; Name: "{app}\source"
Type: filesandordirs; Name: "{app}\tools"
Type: filesandordirs; Name: "{app}\bootstrap"
Type: filesandordirs; Name: "{app}\ffmpeg"
Type: files; Name: "{app}\install.log"
Type: files; Name: "{app}\install-complete.txt"

[Code]
function ShouldInstallRuntime(): Boolean;
var
  I: Integer;
begin
  Result := True;
  for I := 1 to ParamCount do
  begin
    if CompareText(ParamStr(I), '/SKIPRUNTIME') = 0 then
    begin
      Result := False;
      exit;
    end;
  end;
end;

function RuntimeReady(): Boolean;
begin
  Result :=
    FileExists(ExpandConstant('{app}\install-complete.txt')) and
    FileExists(ExpandConstant('{app}\runtime\Scripts\pythonw.exe'));
end;

procedure VerifyRuntime();
begin
  if RuntimeReady() then
    exit;

  MsgBox(
    'AI 런타임 설치에 실패했습니다.' + #13#10 + #13#10 +
    '오류 내용은 다음 파일에 저장되었습니다:' + #13#10 +
    ExpandConstant('{app}\install.log'),
    mbError,
    MB_OK
  );
  RaiseException('AI 런타임 설치 실패');
end;

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
