"""Advanced raw config editor page."""

from shiny import ui, render
from osmose.schema.registry import ParameterRegistry
from osmose.schema.simulation import SIMULATION_FIELDS
from osmose.schema.species import SPECIES_FIELDS
from osmose.schema.grid import GRID_FIELDS
from osmose.schema.predation import PREDATION_FIELDS
from osmose.schema.fishing import FISHING_FIELDS
from osmose.schema.movement import MOVEMENT_FIELDS
from osmose.schema.ltl import LTL_FIELDS
from osmose.schema.output import OUTPUT_FIELDS
from osmose.schema.bioenergetics import BIOENERGETICS_FIELDS
from osmose.schema.economics import ECONOMICS_FIELDS


def _build_registry():
    reg = ParameterRegistry()
    for fields in [
        SIMULATION_FIELDS,
        SPECIES_FIELDS,
        GRID_FIELDS,
        PREDATION_FIELDS,
        FISHING_FIELDS,
        MOVEMENT_FIELDS,
        LTL_FIELDS,
        OUTPUT_FIELDS,
        BIOENERGETICS_FIELDS,
        ECONOMICS_FIELDS,
    ]:
        for f in fields:
            reg.register(f)
    return reg


def advanced_ui():
    registry = _build_registry()
    categories = ["all"] + registry.categories()

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
                    "adv_category", "Category", choices={c: c.title() for c in categories}
                ),
                ui.input_text("adv_search", "Search parameters", placeholder="Type to filter..."),
                ui.p(
                    f"Total parameters in registry: {len(registry.all_fields())}",
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


def advanced_server(input, output, session):
    registry = _build_registry()

    @render.ui
    def param_table():
        category = input.adv_category()
        search = input.adv_search().lower() if input.adv_search() else ""

        if category == "all":
            fields = registry.all_fields()
        else:
            fields = registry.fields_by_category(category)

        if search:
            fields = [
                f
                for f in fields
                if search in f.key_pattern.lower() or search in f.description.lower()
            ]

        if not fields:
            return ui.div("No parameters match your filter.", style="padding: 20px; color: #999;")

        rows = []
        for f in fields[:100]:  # Limit to 100 for performance
            rows.append(
                ui.tags.tr(
                    ui.tags.td(f.key_pattern, style="font-family: monospace; font-size: 12px;"),
                    ui.tags.td(f.param_type.value),
                    ui.tags.td(str(f.default) if f.default is not None else "-"),
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
                        ui.tags.th("Default"),
                        ui.tags.th("Category"),
                        ui.tags.th("Description"),
                    )
                ),
                ui.tags.tbody(*rows),
                class_="table table-striped table-hover table-sm",
            ),
            style="max-height: 600px; overflow-y: auto;",
        )
