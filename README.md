# OSMOSE Python Interface

Python orchestration layer and Shiny web interface for the [OSMOSE](https://osmose-model.org/) marine ecosystem simulator. Replaces the R package while keeping the Java engine unchanged.

## Features

- **Schema-driven parameter system** — 181 parameters defined once, UI auto-generated from metadata
- **Config I/O** — read/write OSMOSE's native `.csv`/`.properties` format with auto-detected separators and recursive sub-file loading
- **Async Java runner** — execute OSMOSE simulations with real-time progress streaming
- **Results reader** — CSV and NetCDF output parsing via xarray (biomass, diet, spatial maps, mortality)
- **Calibration** — multi-objective optimization (pymoo NSGA-II), GP surrogate model, Sobol sensitivity analysis
- **Scenario management** — save, load, compare, and fork named configurations
- **10-tab Shiny UI** — Setup, Grid, Forcing, Fishing, Movement, Run, Results, Calibration, Scenarios, Advanced

## Quick Start

```bash
# Clone and set up
git clone https://github.com/razinkele/osmose-python.git
cd osmose-python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the app
shiny run app.py --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Requirements

- Python 3.12+
- Java 17+ (for running OSMOSE simulations)

## Docker

```bash
docker build -t osmose-python .
docker run -p 8000:8000 osmose-python
```

Place `osmose.jar` in `osmose-java/` before building.

## Project Structure

```
osmose/                  Core library (usable without Shiny)
  schema/                Parameter definitions + registry
  config/                Config reader/writer
  calibration/           pymoo, GP surrogate, SALib sensitivity
  runner.py              Async Java subprocess manager
  results.py             CSV/NetCDF output reader
  scenarios.py           Save/load/compare/fork
ui/                      Shiny web interface
  pages/                 One module per tab
  components/            Reusable widgets (param form)
  theme.py               Shinyswatch superhero theme
data/
  examples/              Bay of Biscay example config (3 species)
tests/                   132 tests
```

## Testing

```bash
pytest                   # run all tests
pytest -v -k test_name   # run specific test
ruff check .             # lint
ruff format .            # format
```

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Shiny for Python + shinyswatch |
| Visualization | plotly |
| NetCDF | xarray, netCDF4 |
| Calibration | pymoo (NSGA-II), scikit-learn (GP) |
| Sensitivity | SALib |
| Config | pandas, jinja2 |
| Deployment | Docker (eclipse-temurin JRE + Python 3.12) |

## License

[MIT](LICENSE)
