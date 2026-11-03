# Build on the target OS; models and credentials remain external user selections.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

project = Path(SPECPATH).parent
datas = []
binaries = []
hiddenimports = ["contextshift.desktop.window", "contextshift.desktop.worker"]
excluded = []
if os.environ.get("CONTEXTSHIFT_BUNDLE_SPEECH") == "1":
    for package in ("funasr", "modelscope", "torchaudio", "sounddevice", "pywebrtc_audio"):
        package_datas, package_binaries, package_imports = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_imports
else:
    excluded = ["funasr", "modelscope", "torch", "torchaudio", "sounddevice", "numpy", "librosa"]

analysis = Analysis(
    [str(project / "contextshift_gui.py")], pathex=[str(project / "src")],
    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=excluded, noarchive=False,
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive, analysis.scripts, [], exclude_binaries=True, name="ContextShift",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
collection = COLLECT(executable, analysis.binaries, analysis.datas, strip=False,
                     upx=False, name="ContextShift")
if sys.platform == "darwin":
    app = BUNDLE(
        collection, name="ContextShift.app", bundle_identifier="io.contextshift.desktop",
        info_plist={
            "NSMicrophoneUsageDescription": "ContextShift 在本地转写麦克风发言，作为实时回答的上下文。",
            "NSHighResolutionCapable": True,
        },
    )
