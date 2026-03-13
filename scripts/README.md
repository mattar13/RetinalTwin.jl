# RetinalTwin CLI Scripts

Command-line scripts for running RetinalTwin simulations from the terminal or from other languages (e.g. Python).

## Setup

Make sure the RetinalTwin project environment is activated and dependencies are installed:

```bash
cd RetinalTwin.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.add("ArgParse")'
```

## run_simulation.jl

Runs an ERG simulation from 4 input files and writes results as CSV.

### Required Input Files

| File | Description |
|------|-------------|
| `structure.json` | Column architecture — which cell types and how they connect |
| `retinal_params.csv` | All biophysical parameters (conductances, time constants, etc.) |
| `stimulus_table.csv` | Flash protocol — one row per sweep with intensity, duration, timing |
| `erg_depth_map.csv` | Depth weighting for computing the field potential from cellular voltages |

All flags are optional. If omitted, defaults point to `examples/inputs/default/`.

### Command-Line Options

```
--structure PATH       Column structure JSON file
--params PATH          Retinal parameters CSV file
--stimulus PATH        Stimulus table CSV file
--depth PATH           ERG depth map CSV file
--tspan START,END      Simulation time span in seconds (default: 0.0,6.0)
--dt STEP              Output time step in seconds (default: 0.01)
--response-window S,E  Window for peak detection in seconds (default: 0.5,1.5)
--outdir DIR           Output directory for result CSVs (default: .)
--verbose              Print progress to stderr
```

### Usage from the Terminal

Run with defaults (uses `examples/inputs/default/`):

```bash
julia --project scripts/run_simulation.jl --verbose --outdir results/
```

Run with custom inputs:

```bash
julia --project scripts/run_simulation.jl \
    --structure my_inputs/structure.json \
    --params my_inputs/retinal_params.csv \
    --stimulus my_inputs/stimulus_table.csv \
    --depth my_inputs/erg_depth_map.csv \
    --tspan 0.0,8.0 \
    --dt 0.005 \
    --outdir results/ \
    --verbose
```

### Usage from Python

Julia's default startup compiles packages on first use, which is slow for repeated subprocess calls.
Build a sysimage once to get near-instant startup on every subsequent run:

```bash
# One-time build (~5-10 minutes). Only needs to be re-run when packages change.
julia --project scripts/build_sysimage.jl
```

This produces `retinal_twin.dll` (Windows), `.dylib` (macOS), or `.so` (Linux) in the project root.

```python
import subprocess
import platform
from pathlib import Path
import pandas as pd

RETINAL_TWIN_DIR = Path("path/to/RetinalTwin.jl")

# Pick the right sysimage extension for the platform
ext = {"Windows": "dll", "Darwin": "dylib"}.get(platform.system(), "so")
sysimage = RETINAL_TWIN_DIR / f"retinal_twin.{ext}"

cmd = [
    "julia", "--project",
    *(["--sysimage", str(sysimage)] if sysimage.exists() else []),
    "scripts/run_simulation.jl",
    "--structure", "my_inputs/structure.json",
    "--params",    "my_inputs/retinal_params.csv",
    "--stimulus",  "my_inputs/stimulus_table.csv",
    "--depth",     "my_inputs/erg_depth_map.csv",
    "--tspan",     "0.0,6.0",
    "--dt",        "0.01",
    "--outdir",    "results/",
    "--verbose",
]

result = subprocess.run(cmd, capture_output=True, text=True, cwd=RETINAL_TWIN_DIR)

if result.returncode != 0:
    raise RuntimeError(f"Simulation failed:\n{result.stderr}")

# Read the output CSVs
traces = pd.read_csv(RETINAL_TWIN_DIR / "results" / "erg_traces.csv")
peaks  = pd.read_csv(RETINAL_TWIN_DIR / "results" / "peak_amplitudes.csv")
```

### Output Files

| File | Format | Description |
|------|--------|-------------|
| `erg_traces.csv` | `time, sweep_1, sweep_2, ...` | Simulated ERG waveform for each stimulus sweep |
| `peak_amplitudes.csv` | `sweep, intensity, peak_amplitude` | Peak a-wave amplitude per sweep |
