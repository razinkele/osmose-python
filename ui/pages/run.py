"""Run control page - execute OSMOSE simulations."""

from shiny import ui, reactive, render


def run_ui():
    return ui.page_fluid(
        ui.layout_columns(
            # Left: Run controls
            ui.card(
                ui.card_header("Run Configuration"),
                ui.input_text("jar_path", "OSMOSE JAR path", value="osmose-java/osmose.jar"),
                ui.input_text(
                    "java_opts", "Java options", value="-Xmx2g", placeholder="-Xmx4g -Xms1g"
                ),
                ui.input_text_area(
                    "param_overrides", "Parameter overrides (key=value, one per line)", rows=4
                ),
                ui.hr(),
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run", "Start Run", class_="btn-success btn-lg w-100"
                    ),
                    ui.input_action_button(
                        "btn_cancel", "Cancel", class_="btn-danger btn-lg w-100"
                    ),
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
        ),
    )


def run_server(input, output, session):
    run_log = reactive.value([])
    status = reactive.value("Idle")

    @render.text
    def run_status():
        return status.get()

    @render.ui
    def run_console():
        lines = run_log.get()
        text = "\n".join(lines[-100:]) if lines else "No output yet. Click 'Start Run' to begin."
        return ui.tags.pre(
            text,
            style="background: #111; color: #0f0; height: 500px; overflow-y: auto; "
            "padding: 12px; border-radius: 6px; font-family: 'Courier New', monospace; "
            "font-size: 13px; white-space: pre-wrap;",
        )
