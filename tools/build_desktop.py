"""Build a standalone app, or a lightweight Mac launcher for this checkout."""

import argparse
import importlib.util
import os
import plistlib
import shlex
import subprocess
import sys
from pathlib import Path


def build_launcher(project: Path, output: Path, python: Path) -> Path:
    """A double-clickable .app bound to an existing installed environment."""

    app = output / "ContextShift.app"
    executable = app / "Contents/MacOS/ContextShift"
    executable.parent.mkdir(parents=True, exist_ok=True)
    (app / "Contents/Resources").mkdir(exist_ok=True)
    plist = {
        "CFBundleName": "ContextShift",
        "CFBundleDisplayName": "ContextShift",
        "CFBundleIdentifier": "io.contextshift.desktop",
        "CFBundleExecutable": "ContextShift",
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1",
        "NSMicrophoneUsageDescription": "ContextShift 在本地转写麦克风发言，作为实时回答的上下文。",
        "NSHighResolutionCapable": True,
    }
    with (app / "Contents/Info.plist").open("wb") as stream:
        plistlib.dump(plist, stream)
    executable.write_text(
        "#!/bin/sh\nset -e\n"
        'mkdir -p "$HOME/Library/Logs/ContextShift"\n'
        + f"cd {shlex.quote(str(project))}\n"
        + f"exec {shlex.quote(str(python))} -m contextshift.desktop.app "
        + '>>"$HOME/Library/Logs/ContextShift/desktop.log" 2>&1\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 ContextShift 桌面应用")
    parser.add_argument(
        "--launcher", action="store_true", help="生成依赖当前项目环境的轻量 macOS .app"
    )
    parser.add_argument(
        "--without-speech",
        action="store_true",
        help="仅用于界面打包检查，不包含语音依赖",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if importlib.util.find_spec("PySide6") is None:
        parser.error("请先运行 uv sync --extra desktop --extra speech --extra bundle")
    if args.launcher:
        print(build_launcher(project, project / "dist", Path(sys.executable)))
        return
    if importlib.util.find_spec("PyInstaller") is None:
        parser.error("缺少打包依赖，请安装 --extra bundle")
    if not args.without_speech:
        missing = [
            name
            for name in ("funasr", "sounddevice", "torchaudio")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            parser.error(
                "缺少语音依赖：" + ", ".join(missing) + "。安装 --extra speech 后重试。"
            )
    env = dict(os.environ)
    env["CONTEXTSHIFT_BUNDLE_SPEECH"] = "0" if args.without_speech else "1"
    env.setdefault("PYINSTALLER_CONFIG_DIR", str(project / ".pyinstaller-cache"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(project / "dist"),
            "--workpath",
            str(project / "build"),
            str(project / "packaging/contextshift.spec"),
        ],
        cwd=project,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
