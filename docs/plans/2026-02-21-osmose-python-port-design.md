# OSMOSE Python Port — Design Document

**Date:** 2026-02-21
**Status:** Approved
**Architecture:** Monolithic Shiny for Python application with schema-driven config system

---

## Context

OSMOSE (Object-oriented Simulator of Marine Ecosystems) is a multispecies individual-based model with a Java computation engine, currently wrapped by an R package. This project replaces the R layer with a Python orchestration layer and Shiny web interface, keeping the Java engine unchanged.

**Target:** Server-deployed project deliverable for non-technical stakeholders.
**Scope:** Full ~200+ parameter coverage, calibration from start, scenario management.

## Architecture Decision

**Option 1 (chosen): Python Orchestrator** — Keep Java engine, replace R layer with Python + Shiny. The Java engine handles all heavy computation. Python handles configuration, execution, output reading, calibration, and visualization.

Rejected alternatives:
- Full Python rewrite of the engine (months–years, high risk)
- JPype/JNI hybrid (fragile, hard to maintain)
- FastAPI + Shiny microservice (overkill for single-consumer UI)
- Notebook-first (poor fit for non-technical stakeholders)

## Project Structure

```
osmose-python/
├── app.py                          # Shiny entry point
├── pyproject.toml                  # Project metadata & dependencies
├── Dockerfile                      # Containerized deployment (Java + Python)
│
├── osmose/                         # Core library (usable without Shiny)
│   ├── __init__.py
│   ├── schema/                     # Parameter schema definitions
│   │   ├── __init__.py
│   │   ├── base.py                 # Schema base classes & field types
│   │   ├── simulation.py           # Simulation parameters
│   │   ├── species.py              # Species parameters (growth, repro, mortality)
│   │   ├── grid.py                 # Grid/spatial configuration
│   │   ├── predation.py            # Predation & accessibility matrix
│   │   ├── fishing.py              # Fishing mortality & fisheries module
│   │   ├── movement.py             # Spatial distribution maps
│   │   ├── ltl.py                  # Lower trophic level / plankton
│   │   ├── output.py               # Output configuration (115+ flags)
│   │   ├── bioenergetics.py        # Bioenergetics module params
│   │   ├── economics.py            # Economic module params
│   │   └── registry.py             # Central parameter registry
│   │
│   ├── config/                     # Config file I/O
│   │   ├── __init__.py
│   │   ├── reader.py               # Parse existing OSMOSE .properties/.csv configs
│   │   ├── writer.py               # Generate OSMOSE config files from schema
│   │   ├── validator.py            # Cross-parameter validation
│   │   └── templates/              # Jinja2 templates for config generation
│   │
│   ├── runner.py                   # Java engine subprocess manager (async)
│   ├── results.py                  # NetCDF output reader (xarray)
│   ├── scenarios.py                # Scenario save/load/compare/fork
│   │
│   └── calibration/                # Calibration engine
│       ├── __init__.py
│       ├── problem.py              # OSMOSE as optimization problem (pymoo)
│       ├── objectives.py           # Objective functions (biomass RMSE, diet distance)
│       ├── surrogate.py            # GP surrogate model for emulation
│       └── sensitivity.py          # Sensitivity analysis (SALib)
│
├── ui/                             # Shiny UI modules
│   ├── __init__.py
│   ├── pages/
│   │   ├── setup.py                # Species & simulation setup
│   │   ├── grid.py                 # Grid configuration & map preview
│   │   ├── forcing.py              # Plankton/LTL & environmental forcing
│   │   ├── fishing.py              # Fishing configuration
│   │   ├── movement.py             # Spatial distribution maps
│   │   ├── run.py                  # Run control, progress, logs
│   │   ├── results.py              # Output visualization dashboard
│   │   ├── calibration.py          # Calibration panel
│   │   ├── scenarios.py            # Scenario management panel
│   │   └── advanced.py             # Raw config editor (all 200+ params)
│   │
│   ├── components/                 # Reusable UI widgets
│   │   ├── species_table.py        # Editable species parameter table
│   │   ├── matrix_editor.py        # Accessibility/diet matrix editor
│   │   ├── map_viewer.py           # Spatial map viewer (plotly)
│   │   ├── param_form.py           # Auto-generated parameter form from schema
│   │   └── file_upload.py          # NetCDF/CSV file upload handlers
│   │
│   └── theme.py                    # Shinyswatch theme configuration
│
├── data/
│   ├── examples/                   # Example OSMOSE configurations
│   ├── defaults/                   # Default parameter values
│   └── scenarios/                  # Saved scenario storage
│
├── osmose-java/                    # Java JAR files (unchanged)
│   └── osmose.jar
│
└── tests/
    ├── test_schema.py
    ├── test_config_reader.py
    ├── test_config_writer.py
    ├── test_runner.py
    └── test_roundtrip.py           # Config read → write → read equality
```

## Core Design Decisions

### 1. Schema-Driven Parameter System

Every OSMOSE parameter is defined once in `osmose/schema/` with metadata:

```python
@dataclass
class OsmoseField:
    key_pattern: str        # e.g. "species.linf.sp{idx}"
    param_type: ParamType   # FLOAT, INT, STRING, BOOL, FILE_PATH, MATRIX, ENUM
    default: Any = None
    min_val: float | None = None
    max_val: float | None = None
    description: str = ""
    category: str = ""
    unit: str = ""
    choices: list[str] | None = None
    indexed: bool = False   # True = per-species (sp{idx})
    required: bool = True
    advanced: bool = False
```

The UI auto-generates forms from schema metadata. Adding a new OSMOSE parameter = adding one field to the schema.

### 2. Config I/O

**Reader** parses OSMOSE's native format:
- Auto-detects separator (=, ;, comma, tab, colon)
- Recursively follows `osmose.configuration.*` references
- Returns flat dict with lowercase keys
- Maps flat keys to structured schema objects

**Writer** generates the full OSMOSE config directory:
- Master file (`osm_all-parameters.csv`) with sub-file references
- Category-specific files (`osm_param-species.csv`, etc.)
- Matrix CSVs (accessibility, movement maps, grid mask)
- Seasonality CSVs (reproduction, fishing)

### 3. Scenario Management

Scenarios are self-contained snapshots:
```
data/scenarios/<name>/
├── scenario.json       # Metadata + all parameter values
├── forcing/            # Copied forcing files
└── maps/               # Copied movement maps
```

Operations: save, load, list, compare (diff), fork (branch from existing).

### 4. Runner

Async subprocess execution of:
```
java [opts] -jar osmose.jar config.csv [-Pkey=val ...]
```
With stdout/stderr streaming to UI progress console.

### 5. Results Reader

xarray-based NetCDF reader exposing:
- Time series: biomass, abundance, yield, SSB
- Distributions: by size, age, weight, trophic level
- Trophic: diet composition matrix, predation pressure
- Spatial: gridded biomass/abundance maps
- Mortality: breakdown by source (predation, fishing, starvation, natural)

### 6. Calibration

Two-track approach:
1. **Direct optimization** (pymoo NSGA-II): Run OSMOSE for each candidate in parallel (isolated temp dirs). Multi-objective: biomass RMSE + diet matrix distance.
2. **Surrogate model** (scikit-learn GP): Latin hypercube sampling → build GP emulator → optimize on emulator → validate on real OSMOSE runs. Cuts calibration from weeks to hours.

## UI Layout

9-tab `page_navbar` with shinyswatch `superhero` theme (dark blue-grey, orange accents):

| Tab | Purpose |
|-----|---------|
| Setup | Species count, names, growth/reproduction/mortality parameters |
| Grid & Maps | Grid dimensions, lat/lon, mask upload, movement map editor |
| Forcing | LTL/plankton config, NetCDF uploads, temperature/oxygen forcing |
| Fishing | Fishing mortality, fisheries module, selectivity, MPAs |
| Run | Start/stop, progress console, Java log streaming, run history |
| Results | Biomass timeseries, diet matrices, spatial maps, distributions (plotly) |
| Calibration | Objectives, free params, algorithm selection, Pareto front viewer |
| Scenarios | Save/load/compare/fork named configurations |
| Advanced | Raw searchable parameter table for all 200+ params |

## Technology Stack

| Component | Library |
|-----------|---------|
| UI Framework | Shiny for Python |
| Theme | shinyswatch (superhero) |
| Config generation | pandas, jinja2 |
| Java execution | asyncio subprocess |
| NetCDF reading | xarray, netCDF4 |
| Visualization | plotly |
| Spatial maps | plotly (choropleth/heatmaps) |
| Calibration | pymoo (NSGA-II), scikit-learn (GP surrogate) |
| Sensitivity | SALib |
| Deployment | Docker (eclipse-temurin JRE + Python 3.12) |

## OSMOSE Configuration Reference

### Parameter Naming Convention

All parameters use dot-separated hierarchical keys, lowercase. Species-indexed parameters use `.sp{N}` suffix (0-based). Fishery-indexed use `.fsh{N}`. Resource-indexed continue numbering after focal species.

### Key Parameter Categories (~200+ total)

1. **Simulation** (~15 params): `simulation.time.ndtperyear`, `simulation.nspecies`, `simulation.ncpu`, etc.
2. **Species** (~30 params per species): growth (VB/Gompertz), length-weight, reproduction, life history, population init
3. **Grid** (~10 params): regular grid or NetCDF grid, mask
4. **Predation** (~10 params + matrix): ingestion rate, size ratios, accessibility matrix
5. **Mortality** (~15 params per species): natural, starvation, additional (constant or time-varying)
6. **Fishing** (~20 params + fisheries module): F rates, catches, selectivity, seasonality, MPAs
7. **Movement** (~5 params per map): species maps by age/season/year
8. **LTL/Resources** (~10 params per group): plankton biomass, size range, trophic level
9. **Output** (115+ boolean flags + settings): which outputs to produce
10. **Bioenergetics** (~15 params per species): maintenance energy, assimilation, foraging
11. **Economics** (~5 params): economic output stages

### Config File Formats

| Type | Format | Example |
|------|--------|---------|
| Parameters | Key-value text (.csv) | `species.linf.sp0 ; 164.50` |
| Movement maps | CSV matrix (nline × ncolumn) | Probability values 0–1 |
| Grid mask | CSV matrix (nline × ncolumn) | 0=land, 1=ocean |
| Accessibility | CSV with headers | Predators as columns, prey as rows |
| Seasonality | CSV time series | Monthly spawning/fishing intensity |
| Forcing data | NetCDF (.nc) | Gridded biomass/temperature fields |

## Deployment

Single Docker container with multi-stage build:
- Stage 1: eclipse-temurin:17-jre (Java runtime)
- Stage 2: python:3.12-slim + Java copied from stage 1
- Exposes port 8000 for Shiny server
