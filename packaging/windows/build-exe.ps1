<#
.SYNOPSIS
    Build the Windows installer for the PySide6 front-end.

.DESCRIPTION
    PyInstaller --onedir --windowed from ui_pyside/main.py, then Inno Setup to
    wrap the bundle into one setup.exe.

    The `dml` extra is bundled: onnxruntime-directml is the one execution
    provider that reaches NVIDIA, AMD and Intel alike on Windows, over DirectX
    12, so the fat Windows package is the DirectML one rather than a CPU
    runtime. core/backends/onnx_backend.py already sets enable_mem_pattern=False
    and ORT_SEQUENTIAL for DirectML, because it refuses the defaults.

    The .iss script is generated from a here-string rather than kept as a file,
    so the output name, the source directory and the version have exactly one
    home: the parameters of this script. A checked-in .iss with a version typed
    into it is how an installer ends up shipping the wrong one.

    The dependency list is read out of pyproject.toml by the same Python that
    runs PyInstaller, rather than parsed out of the TOML with a regular
    expression. PowerShell has no TOML parser, and a regex over pyproject.toml is
    a parser that fails silently on the day someone reformats the file.

.PARAMETER Version
    The version in the file name and in the installer. Defaults to the `version`
    in pyproject.toml, which is where it lives.

.PARAMETER OutDir
    Where the .exe is written. Defaults to <repo>\dist.

.PARAMETER InnoVersion
    The Inno Setup release this build expects. Used only in the failure message:
    this script installs nothing, so a build machine's package inventory is
    never changed behind its back.

.PARAMETER InnoPath
    An explicit path to iscc.exe. Overrides the PATH lookup and InnoVersion.

.PARAMETER SkipInstall
    Reuse the site directory already under build\windows, skipping the
    dependency install.

.PARAMETER DryRun
    Print every command and every generated file without running anything.

.EXAMPLE
    .\packaging\windows\build-exe.ps1
    .\packaging\windows\build-exe.ps1 -Version 0.1.0 -DryRun
#>

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path
$ScriptName = Split-Path -Leaf $MyInvocation.MyCommand.Path

# PyInstaller is a build tool, not a runtime dependency, and its version is
# pinned so two builds of one commit produce the same bundle.
$PyInstallerPin = 'pyinstaller==6.22.3'

function Stop-WithError {
    param([Parameter(Mandatory = $true)][string] $Message)
    throw "$ScriptName : $Message"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string] $FilePath,
        [string[]] $Arguments = @()
    )
    if ($DryRun) {
        $parts = @($FilePath) + $Arguments
        $rendered = ($parts | ForEach-Object {
                if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
            }) -join ' '
        Write-Host "+ $rendered"
        return
    }
    Write-Host ("+ " + $FilePath + " " + ($Arguments -join ' '))
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "$FilePath exited with $LASTEXITCODE"
    }
}

# Write a generated file as UTF-8 without a BOM. Set-Content -Encoding UTF8 adds
# one in Windows PowerShell 5.1, and the Inno Setup preprocessor stops on it.
function Write-GeneratedFile {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string[]] $Lines
    )
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($Path, $Lines, $Utf8NoBom)
}

# --- parameters --------------------------------------------------------------

$Pyproject = Join-Path $RepoRoot 'pyproject.toml'
if (-not (Test-Path $Pyproject)) {
    Stop-WithError "no pyproject.toml at $Pyproject"
}

$Interpreter = Get-Command 'python' -ErrorAction SilentlyContinue
if (-not $Interpreter) {
    Stop-WithError 'python is not on PATH. Install Python 3.11 or newer and enable the "Add python.exe to PATH" option.'
}
$BasePython = $Interpreter.Source

if (-not $Version) {
    # `version = "..."` appears twice in pyproject.toml: once in [project] and
    # once in [tool.briefcase]. The first is the application's version.
    $VersionLine = Select-String -Path $Pyproject -Pattern '^version\s*=\s*"([^"]+)"' |
        Select-Object -First 1
    if (-not $VersionLine) {
        Stop-WithError "no version = ""..."" line in $Pyproject"
    }
    $Version = $VersionLine.Matches[0].Groups[1].Value
}
if (-not $OutDir) {
    $OutDir = Join-Path $RepoRoot 'dist'
}

$AppName = 'upscaler'
$BundleName = 'Upscaler'
$Publisher = 'AliRezaei-Code'
$SetupName = "upscaler_${Version}_windows-amd64-setup.exe"
$SetupPath = Join-Path $OutDir $SetupName

$BuildDir = Join-Path $RepoRoot 'build\windows'
$SiteDir = Join-Path $BuildDir 'site'
$DistDir = Join-Path $BuildDir 'dist'
$WorkDir = Join-Path $BuildDir 'work'
$SpecDir = Join-Path $BuildDir 'spec'
$IssPath = Join-Path $BuildDir "${AppName}.iss"
$IconPath = Join-Path $RepoRoot 'packaging\icon.png'
$LicensePath = Join-Path $RepoRoot 'LICENSE'

if (-not (Test-Path $IconPath)) {
    Stop-WithError "missing $IconPath"
}
if (-not (Test-Path $LicensePath)) {
    Stop-WithError "missing $LicensePath; the Qt licence has to travel with the binary"
}

# --- the dependency list -----------------------------------------------------

$RequirementsScript = @'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    project = tomllib.load(handle)

names = list(project["project"]["dependencies"])
names += list(project["project"]["optional-dependencies"][sys.argv[2]])
for requirement in names:
    name = re.split(r"[\s\[<>=!~;(]", requirement, maxsplit=1)[0]
    # toga is dropped for the same reason the Linux build drops it: the frozen
    # executable is the Qt front-end and never imports it.
    if not name.lower().startswith("toga"):
        print(requirement)
'@

function Get-Requirements {
    $Query = Join-Path $BuildDir 'read_requirements.py'
    if ($DryRun) {
        return @('<read from pyproject.toml at run time>')
    }
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    [System.IO.File]::WriteAllText($Query, $RequirementsScript, (New-Object System.Text.UTF8Encoding($false)))
    $Output = & $BasePython $Query $Pyproject 'dml'
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "reading the dependency list from $Pyproject exited with $LASTEXITCODE"
    }
    $Lines = @($Output | Where-Object { $_.Trim() -ne '' })
    if ($Lines.Count -eq 0) {
        Stop-WithError "pyproject.toml yielded no requirements; the [dml] extra is required"
    }
    return $Lines
}

# --- prerequisites -----------------------------------------------------------

function Resolve-InnoCompiler {
    if ($InnoPath) {
        if (Test-Path $InnoPath) {
            return (Resolve-Path $InnoPath).Path
        }
        Stop-WithError "-InnoPath was $InnoPath, which does not exist"
    }
    $OnPath = Get-Command 'iscc.exe' -ErrorAction SilentlyContinue
    if ($OnPath) {
        return $OnPath.Source
    }
    $ProgramFilesX86 = [Environment]::GetFolderPath('ProgramFilesX86')
    if ($ProgramFilesX86) {
        $Default = Join-Path $ProgramFilesX86 'Inno Setup 6\ISCC.exe'
        if (Test-Path $Default) {
            return $Default
        }
    }
    Stop-WithError ("iscc.exe is not on PATH, and there is no ISCC.exe under " +
        "$ProgramFilesX86\Inno Setup 6. Install Inno Setup $InnoVersion from " +
        'https://jrsoftware.org/isdl.php and re-run, or pass -InnoPath with ' +
        'the full path to ISCC.exe.')
}

$Iscc = Resolve-InnoCompiler

# --- the Inno Setup script ---------------------------------------------------

function New-InnoScript {
    param(
        [Parameter(Mandatory = $true)][string] $OutputBase,
        [Parameter(Mandatory = $true)][string] $SourceDir,
        [Parameter(Mandatory = $true)][string] $LicenseFile
    )
    $Lines = @()
    $Lines += '; Generated by packaging/windows/build-exe.ps1 -- do not edit.'
    $Lines += '; Regenerate it by re-running the build; every value below comes from the build parameters.'
    $Lines += ''
    $Lines += '#define MyAppName "' + $BundleName + '"'
    $Lines += '#define MyAppVersion "' + $Version + '"'
    $Lines += '#define MyAppExeName "' + $AppName + '.exe"'
    $Lines += '#define MyAppPublisher "' + $Publisher + '"'
    $Lines += '#define MyAppSourceDir "' + $SourceDir + '"'
    $Lines += '#define MyAppOutputBase "' + $OutputBase + '"'
    $Lines += ''
    $Lines += '[Setup]'
    $Lines += 'AppId = {{A6F0B1E2-7C4D-4E2A-9E3B-5C1D0A9F0001}'
    $Lines += 'AppName = {#MyAppName}'
    $Lines += 'AppVersion = {#MyAppVersion}'
    $Lines += 'AppPublisher = {#MyAppPublisher}'
    $Lines += 'AppPublisherURL = https://github.com/AliRezaei-Code/upscaler'
    $Lines += 'DefaultDirName = {autopf}\$MyAppName'
    $Lines += 'DefaultGroupName = $MyAppName'
    $Lines += 'OutputDir = ' + $OutDir
    $Lines += 'OutputBaseFilename = {#MyAppOutputBase}'
    $Lines += 'Compression = lzma2/ultra64'
    $Lines += 'SolidCompression = yes'
    $Lines += 'LicenseFile = ' + $LicenseFile
    $Lines += 'WizardStyle = modern'
    $Lines += 'ArchitecturesAllowed = x64compatible'
    $Lines += 'ArchitecturesInstallIn64BitMode = x64compatible'
    $Lines += 'PrivilegesRequired = admin'
    $Lines += 'DisableProgramGroupPage = yes'
    $Lines += ''
    $Lines += '[Languages]'
    $Lines += 'Name: "english"; MessagesFile: "compiler:Default.isl"'
    $Lines += ''
    $Lines += '[Tasks]'
    $Lines += 'Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"'
    $Lines += ''
    $Lines += '[Files]'
    $Lines += 'Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs'
    $Lines += ''
    $Lines += '[Icons]'
    $Lines += 'Name: "{group}\$MyAppName"; Filename: "{app}\$MyAppExeName"'
    $Lines += 'Name: "{group}\Uninstall $MyAppName"; Filename: "{uninstallexe}"'
    $Lines += 'Name: "{autodesktop}\$MyAppName"; Filename: "{app}\$MyAppExeName"; Tasks: desktopicon'
    $Lines += ''
    $Lines += '[Run]'
    $Lines += 'Filename: "{app}\$MyAppExeName"; Description: "Launch $MyAppName"; Flags: nowait postinstall skipifsilent'
    return $Lines
}

# --- the build ---------------------------------------------------------------

Write-Host "app         : $BundleName $Version"
Write-Host "python      : $BasePython"
Write-Host "iscc        : $Iscc"
Write-Host "build tree  : $BuildDir"
Write-Host "artefact    : $SetupPath"
if ($DryRun) { Write-Host 'DRY RUN     : nothing below is executed' }
Write-Host ''

# `@()` because a PowerShell function that returns a single element unwraps to
# a scalar, and `'a' + 'b'` concatenates two strings instead of building an
# argument list.
$Requirements = @(Get-Requirements)
if ($DryRun) {
    Write-Host ("requirements: read from pyproject.toml at run time (base minus toga*, plus the [dml] extra)")
} else {
    Write-Host ("requirements: " + ($Requirements -join ' '))
}

# The bundle is frozen from a build venv that holds only PyInstaller plus the
# dependency set, for the reason the Linux build documents: the flavour has to
# be exact, and a Windows build agent's interpreter is not a statement about
# which execution provider this installer should ship.
$VenvDir = Join-Path $BuildDir 'venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
if (-not $SkipInstall) {
    if ($DryRun) {
        Invoke-Checked -FilePath $BasePython -Arguments @('-m', 'venv', $VenvDir)
        Invoke-Checked -FilePath $VenvPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--quiet', '--upgrade', $PyInstallerPin)
    } elseif (-not (Test-Path $VenvPython)) {
        Invoke-Checked -FilePath $BasePython -Arguments @('-m', 'venv', $VenvDir)
        Invoke-Checked -FilePath $VenvPython -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--quiet', '--upgrade', $PyInstallerPin)
    }
    Invoke-Checked -FilePath $VenvPython -Arguments (@('-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', '--target', $SiteDir) + $Requirements)
} else {
    Write-Host 'step 1      : skipped, reusing the site directory'
}

# DirectML is the one provider that covers every GPU vendor on Windows, so this
# is the check that the build is what it claims: an onnxruntime with no
# DmlExecutionProvider in it is the CPU wheel under another name, and the
# installer would then ship an application that cannot use a GPU.
$ProviderScript = @'
import onnxruntime

providers = onnxruntime.get_available_providers()
print("onnxruntime:", onnxruntime.__file__)
print("providers  :", ", ".join(providers))
if "DmlExecutionProvider" not in providers:
    raise SystemExit(
        "the installed ONNX Runtime has no DmlExecutionProvider; the dml extra "
        "did not install, and the installer would ship a CPU-only application"
    )
'@
if ($DryRun) {
    Write-Host "+ $VenvPython check_providers.py"
} else {
    $ProviderPath = Join-Path $BuildDir 'check_providers.py'
    [System.IO.File]::WriteAllText($ProviderPath, $ProviderScript, (New-Object System.Text.UTF8Encoding($false)))
    Invoke-Checked -FilePath $VenvPython -Arguments @($ProviderPath)
}

$PyiArgs = @(
    '--noconfirm', '--clean',
    '--name', $AppName,
    '--distpath', $DistDir,
    '--workpath', $WorkDir,
    '--specpath', $SpecDir,
    '--paths', $SiteDir,
    '--paths', $RepoRoot,
    '--add-data', ((Join-Path $RepoRoot 'models') + ';models'),
    '--icon', $IconPath,
    '--windowed',
    '--collect-all', 'spandrel',
    '--collect-all', 'cv2',
    '--collect-all', 'selectolax',
    '--hidden-import', 'onnxruntime',
    '--hidden-import', 'spandrel',
    '--hidden-import', 'cv2',
    '--hidden-import', 'core.app',
    '--hidden-import', 'core.pipeline',
    '--hidden-import', 'core.backends.onnx_backend',
    '--hidden-import', 'core.backends.torch_backend',
    '--hidden-import', 'ui_pyside.window',
    '--exclude-module', 'tkinter',
    '--exclude-module', 'unittest',
    '--exclude-module', 'pydoc',
    '--exclude-module', 'doctest',
    '--exclude-module', 'pdb',
    '--exclude-module', 'lib2to3',
    (Join-Path $RepoRoot 'ui_pyside\main.py')
)
Invoke-Checked -FilePath $VenvPython -Arguments (@('-m', 'PyInstaller') + $PyiArgs)

$Iss = New-InnoScript -OutputBase $SetupName -SourceDir (Join-Path $DistDir $AppName) -LicenseFile $LicensePath
if ($DryRun) {
    Write-Host "+ write $Iss"
    foreach ($Line in $Iss) { Write-Host "  $Line" }
} else {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Write-GeneratedFile -Path $IssPath -Lines $Iss
}

Invoke-Checked -FilePath $Iscc -Arguments @($IssPath)

if ($DryRun) {
    Write-Host ''
    Write-Host "artefact: $SetupPath"
    Write-Host 'size:     not built (dry run)'
    exit 0
}

if (-not (Test-Path $SetupPath)) {
    Stop-WithError "Inno Setup finished but produced no $SetupPath"
}
$Size = (Get-Item $SetupPath).Length
$Readable = if ($Size -ge 1GB) {
    '{0:0.00} GiB ({1} bytes)' -f ($Size / 1GB), $Size
} else {
    '{0:0} MiB ({1} bytes)' -f ($Size / 1MB), $Size
}
Write-Host ''
Write-Host "artefact: $SetupPath"
Write-Host "size:     $Readable"
if ($Size -ge 2147483648) {
    Write-Warning 'This installer is 2 GiB or more. GitHub rejects any release asset that size, so it cannot be attached to a release.'
}
