from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags

CRATE_NAME = "cambi"


class CPU(NamedTuple):
    name: str
    suffix: str
    disable: str = ""


class Target(NamedTuple):
    prefix: str
    ext: str
    cpus: list[CPU] | None = None


x86_64_CPUS = [
    CPU("x86-64", ""),
    CPU("haswell", ".avx2"),
    CPU("znver4", ".zn4", disable="sse4a"),
]

TARGETS = {
    "linux": {
        "x86_64": Target("lib", ".so", x86_64_CPUS),
        "aarch64": Target("lib", ".so"),
    },
    "windows": {"amd64": Target("", ".dll", x86_64_CPUS)},
    "darwin": {
        "x86_64": Target("lib", ".dylib", x86_64_CPUS[:-1]),
        "arm64": Target("lib", ".dylib"),
    },
}


def get_host_target_triple() -> str:
    """Query rustc for the host target triple."""

    result = subprocess.run(["rustc", "-vV"], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Could not determine host target triple from `rustc -vV`")


class CustomHook(BuildHookInterface[Any]):
    target_dir = Path("vapoursynth/plugins/cambi")

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{next(tags.platform_tags())}"

        uname = platform.uname()
        target = TARGETS[uname.system.lower()][uname.machine.lower()]

        self.target_dir.mkdir(parents=True, exist_ok=True)

        if os.environ.get("CAMBI_MULTIPLE_TARGET", "").lower() in {"true", "1"} and target.cpus:
            print("Building multiple targets...", file=sys.stderr)

            host_target_triple = get_host_target_triple()

            for cpu in target.cpus:
                rust_flags = [f"-C target-cpu={cpu.name}"]

                if cpu.disable:
                    rust_flags.append(f"-C target-feature=-{cpu.disable}")

                env = os.environ.copy()
                env["RUSTFLAGS"] = " ".join(rust_flags)
                built_target_dir = Path(f"target/{cpu.name}")
                subprocess.run(
                    ["cargo", "build", "--release", "--target", host_target_triple, "--target-dir", built_target_dir],
                    check=True,
                    env=env,
                )

                built = built_target_dir / host_target_triple / "release" / f"{target.prefix}{CRATE_NAME}{target.ext}"
                shutil.copy2(built, self.target_dir / Path(built.name).with_suffix(cpu.suffix + built.suffix))
        else:
            env = os.environ.copy()
            env["RUSTFLAGS"] = "-C target-cpu=native"

            subprocess.run(["cargo", "build", "--release"], check=True, env=env)

            built = Path("target") / "release" / f"{target.prefix}{CRATE_NAME}{target.ext}"
            shutil.copy2(built, self.target_dir)

        manifest = self.target_dir / "manifest.vs"
        manifest.write_text(
            f"[VapourSynth Manifest V1]\n{target.prefix}{CRATE_NAME}\n",
            encoding="utf-8",
        )

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        shutil.rmtree(self.target_dir, ignore_errors=True)
