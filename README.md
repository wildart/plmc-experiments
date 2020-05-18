# Sources for experiments

## Installation

- Install Python packages using Conda **environment.yml** configuration.
    -  `conda env create --file environment.yml`
- Add `BoffinStuff` Julia registry by running **install-reg.jl** script.
    - `julia install-reg.jl`
- Instantiate Julia environment from **Project.toml** configuration.
    1.  `julia --project`
    2.  `] instantiate`
