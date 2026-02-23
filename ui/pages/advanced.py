"""Advanced raw config editor page."""

import tempfile
from pathlib import Path

from shiny import reactive, render, ui

from osmose.config.reader import OsmoseConfigReader
from osmose.config.writer import OsmoseConfigWriter
from ui.state import REGISTRY


def advanced_ui():
    categories = ["all"] + REGISTRY.categories()

    return ui.page_fluid(
        ui.layout_columns(
            # Controls
            ui.card(
                ui.card_header("Config I/O"),
                ui.input_file(
                    "import_config", "Import OSMOSE config", accept=[".csv", ".properties"]
                ),
                ui.download_button(
                    "export_config", "Export Current Config", class_="btn-primary w-100"
                ),
            ),
            ui.card(
                ui.card_header("Filters"),
                ui.input_select(
                    "adv_category",
                    "Category",
                    choices={c: c.title() for c in categories},
                ),
                ui.input_text("adv_search", "Search parameters", placeholder="Type to filter..."),
                ui.p(
                    f"Total parameters in registry: {len(REGISTRY.all_fields())}",
                    style="color: #999; font-size: 12px;",
                ),
            ),
            col_widths=[4, 8],
        ),
        ui.card(
            ui.card_header("All Parameters"),
            ui.output_ui("param_table"),
        ),
    )


def advanced_server(input, output, session, state):
    @reactive.effect
    @reactive.event(input.import_config)
    def handle_import():
        file_info = input.import_config()
        if not file_info:
            return
        filepath = Path(file_info[0]["datapath"])
        reader = OsmoseConfigReader()
        loaded = reader.read_file(filepath)
        # Merge into current config
        cfg = dict(state.config.get())
        cfg.update(loaded)
        state.config.set(cfg)

    @render.download(filename="osm_all-parameters.csv")
    def export_config():
        work_dir = Path(tempfile.mkdtemp(prefix="osmose_export_"))
        writer = OsmoseConfigWriter()
        writer.write(state.config.get(), work_dir)
        master = work_dir / "osm_all-parameters.csv"
        return str(master)

    @render.ui
    def param_table():
        category = input.adv_category()
        search = input.adv_search().lower() if input.adv_search() else ""

        if category == "all":
            fields = REGISTRY.all_fields()
        else:
            fields = REGISTRY.fields_by_category(category)

        if search:
            fields = [
                f
                for f in fields
                if search in f.key_pattern.lower() or search in f.description.lower()
            ]

        if not fields:
            return ui.div("No parameters match your filter.", style="padding: 20px; color: #999;")

        # Show current config values
        cfg = state.config.get()

        rows = []
        for f in fields[:100]:
            current_val = cfg.get(f.key_pattern, "-")
            rows.append(
                ui.tags.tr(
                    ui.tags.td(f.key_pattern, style="font-family: monospace; font-size: 12px;"),
                    ui.tags.td(f.param_type.value),
                    ui.tags.td(str(current_val)),
                    ui.tags.td(f.category),
                    ui.tags.td(
                        f.description[:60] + "..." if len(f.description) > 60 else f.description
                    ),
                )
            )

        return ui.tags.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Key"),
                        ui.tags.th("Type"),
                        ui.tags.th("Current Value"),
                        ui.tags.th("Category"),
                        ui.tags.th("Description"),
                    )
                ),
                ui.tags.tbody(*rows),
                class_="table table-striped table-hover table-sm",
            ),
            style="max-height: 600px; overflow-y: auto;",
        )
