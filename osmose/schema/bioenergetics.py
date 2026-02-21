"""Bioenergetics OSMOSE parameter definitions."""

from osmose.schema.base import OsmoseField, ParamType

BIOENERGETICS_FIELDS: list[OsmoseField] = [
    # ── Temperature settings ──────────────────────────────────────────────
    OsmoseField(
        key_pattern="temperature.filename",
        param_type=ParamType.FILE_PATH,
        description="Temperature NetCDF file",
        category="bioenergetics",
    ),
    OsmoseField(
        key_pattern="temperature.varname",
        param_type=ParamType.STRING,
        default="temp",
        description="Temperature variable name in NetCDF file",
        category="bioenergetics",
    ),
    OsmoseField(
        key_pattern="temperature.nsteps.year",
        param_type=ParamType.INT,
        default=12,
        description="Number of temperature time steps per year",
        category="bioenergetics",
    ),
    OsmoseField(
        key_pattern="temperature.factor",
        param_type=ParamType.FLOAT,
        default=1.0,
        description="Multiplicative factor applied to temperature values",
        category="bioenergetics",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="temperature.offset",
        param_type=ParamType.FLOAT,
        default=0.0,
        description="Additive offset applied to temperature values",
        category="bioenergetics",
        advanced=True,
    ),
    OsmoseField(
        key_pattern="temperature.value",
        param_type=ParamType.FLOAT,
        description="Constant temperature value (overrides NetCDF)",
        category="bioenergetics",
        required=False,
    ),
    # ── Per-species bioenergetics parameters ──────────────────────────────
    OsmoseField(
        key_pattern="species.bioen.assimilation.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Assimilation efficiency",
        category="bioenergetics",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="species.bioen.maint.energy.c_m.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Maintenance energy coefficient",
        category="bioenergetics",
        indexed=True,
    ),
    OsmoseField(
        key_pattern="species.beta.sp{idx}",
        param_type=ParamType.FLOAT,
        description="Beta parameter for bioenergetics",
        category="bioenergetics",
        indexed=True,
        advanced=True,
    ),
]
