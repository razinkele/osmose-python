"""Run control page - execute OSMOSE simulations."""

import tempfile
from pathlib import Path

from shiny import ui, reactive, render

from osmose.config.writer import OsmoseConfigWriter
from osmose.runner import OsmoseRunner


def parse_overrides(text: str) -> dict[str, str]:
    """Parse a text area of key=value lines into a dict."""
    result = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


def write_temp_config(config: dict[str, str], output_dir: Path) -> Path:
    """Write config to a directory and return the master file path."""
    writer = OsmoseConfigWriter()
    writer.write(config, output_dir)
    return output_dir / "osm_all-parameters.csv"


def run_ui():
    return ui.layout_columns(
        # Left: Run controls
        ui.card(
            ui.card_header("Run Configuration"),
            ui.input_text("jar_path", "OSMOSE JAR path", value="osmose-java/osmose.jar"),
            ui.input_text("java_opts", "Java options", value="-Xmx2g", placeholder="-Xmx4g -Xms1g"),
            ui.input_text_area(
                "param_overrides", "Parameter overrides (key=value, one per line)", rows=4
            ),
            ui.hr(),
            ui.layout_columns(
                ui.input_action_button("btn_run", "Start Run", class_="btn-success btn-lg w-100"),
                ui.input_action_button("btn_cancel", "Cancel", class_="btn-danger btn-lg w-100"),
                col_widths=[6, 6],
            ),
            ui.hr(),
            ui.h5("Run Status"),
            ui.output_text("run_status"),
        ),
        # Right: Console output
        ui.card(
            ui.card_header("Console Output"),
            ui.output_ui("run_console"),
        ),
        col_widths=[4, 8],
    )


def run_server(input, output, session, state):
    run_log = reactive.value([])
    status = reactive.value("Idle")
    runner_ref = reactive.value(None)

    @reactive.effect
    def sync_jar_path():
        state.jar_path.set(input.jar_path())

    @render.text
    def run_status():
        return status.get()

    @render.ui
    def run_console():
        lines = run_log.get()
        text = "\n".join(lines[-200:]) if lines else "No output yet. Click 'Start Run' to begin."
        return ui.tags.pre(
            text,
            style="background: #111; color: #0f0; height: 500px; overflow-y: auto; "
            "padding: 12px; border-radius: 6px; font-family: 'Courier New', monospace; "
            "font-size: 13px; white-space: pre-wrap;",
        )

    @reactive.effect
    @reactive.event(input.btn_run)
    async def handle_run():
        jar_path = Path(state.jar_path.get())
        if not jar_path.exists():
            status.set(f"Error: JAR not found at {jar_path}")
            return

        status.set("Writing config...")
        run_log.set([])

        # Write config to temp directory
        config = state.config.get()
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_run_"))
        config_path = write_temp_config(config, work_dir)

        # Parse overrides and java opts
        overrides = parse_overrides(input.param_overrides() or "")
        java_opts_text = input.java_opts() or ""
        java_opts = java_opts_text.split() if java_opts_text.strip() else None

        # Create runner
        runner = OsmoseRunner(jar_path=jar_path)
        runner_ref.set(runner)

        status.set("Running...")

        def on_progress(line: str):
            lines = list(run_log.get())
            lines.append(line)
            run_log.set(lines)

        result = await runner.run(
            config_path=config_path,
            output_dir=work_dir / "output",
            java_opts=java_opts,
            overrides=overrides,
            on_progress=on_progress,
        )

        state.run_result.set(result)
        state.output_dir.set(result.output_dir)

        if result.returncode == 0:
            status.set(f"Complete. Output: {result.output_dir}")
        else:
            status.set(f"Failed (exit code {result.returncode})")
            if result.stderr:
                lines = list(run_log.get())
                lines.append(f"--- STDERR ---\n{result.stderr}")
                run_log.set(lines)

    @reactive.effect
    @reactive.event(input.btn_cancel)
    def handle_cancel():
        runner = runner_ref.get()
        if runner:
            runner.cancel()
            status.set("Cancelled")
