"""Manage OSMOSE Java engine execution."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class RunResult:
    """Result of an OSMOSE simulation run."""

    returncode: int
    output_dir: Path
    stdout: str
    stderr: str


class OsmoseRunner:
    """Execute the OSMOSE Java engine and stream progress."""

    def __init__(self, jar_path: Path, java_cmd: str = "java"):
        self.jar_path = jar_path
        self.java_cmd = java_cmd
        self._process: asyncio.subprocess.Process | None = None

    def _build_cmd(
        self,
        config_path: Path,
        output_dir: Path | None = None,
        java_opts: list[str] | None = None,
        overrides: dict[str, str] | None = None,
    ) -> list[str]:
        """Build the command list for executing the OSMOSE engine.

        Subclasses or tests may override this to adjust command construction.
        """
        cmd = [self.java_cmd]
        if java_opts:
            cmd.extend(java_opts)
        cmd.extend(["-jar", str(self.jar_path), str(config_path)])
        if output_dir:
            cmd.append(f"-Poutput.dir.path={output_dir}")
        if overrides:
            for key, value in overrides.items():
                cmd.append(f"-P{key}={value}")
        return cmd

    async def run(
        self,
        config_path: Path,
        output_dir: Path | None = None,
        java_opts: list[str] | None = None,
        overrides: dict[str, str] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> RunResult:
        """Run the OSMOSE engine asynchronously.

        Args:
            config_path: Path to the master OSMOSE config file.
            output_dir: Override for output directory (passed as -P flag).
            java_opts: Extra JVM options (e.g., ["-Xmx4g"]).
            overrides: Extra parameter overrides (passed as -Pkey=value).
            on_progress: Callback for each line of stdout/stderr.

        Returns:
            RunResult with returncode, stdout, stderr.
        """
        cmd = self._build_cmd(config_path, output_dir, java_opts, overrides)

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def read_stream(
            stream: asyncio.StreamReader | None,
            lines_list: list[str],
        ) -> None:
            if stream is None:
                return
            async for line in stream:
                text = line.decode().rstrip()
                lines_list.append(text)
                if on_progress:
                    on_progress(text)

        await asyncio.gather(
            read_stream(self._process.stdout, stdout_lines),
            read_stream(self._process.stderr, stderr_lines),
        )

        await self._process.wait()
        result_output_dir = output_dir or config_path.parent / "output"

        return RunResult(
            returncode=self._process.returncode,
            output_dir=result_output_dir,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
        )

    def cancel(self) -> None:
        """Terminate the running OSMOSE process."""
        if self._process and self._process.returncode is None:
            self._process.terminate()

    @staticmethod
    def get_java_version(java_cmd: str = "java") -> str | None:
        """Check if Java is installed and return version string."""
        try:
            result = subprocess.run(
                [java_cmd, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Java prints version to stderr
            return result.stderr.strip() or result.stdout.strip() or None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
