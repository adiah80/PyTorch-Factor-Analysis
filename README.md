# PyTorch Factor Analysis

## Installation
```bash
git clone https://github.com/adiah80/PyTorch-Factor-Analysis.git  # Clone the repository
source setup_evironment.sh <new_env_name>                         # Create a new environment and install packages
python setup.py install                                           # Setup the package
```
## Usage

Reproducing the above Plots.
```bash
python generate_plots.py
```

Using the package API.
```bash
python example.py --config-idx <config_index>
```

Python API - PseudoCode
```python
from pytorch_factor_analysis import  ...     # Do the imports.

device = torch.device(...)                   # Set the device for PyTorch.
cfg = {...}                                  # Set the run config file.

data = generate_sample_data(...)             # Sample Data for the run.
FA_object = FA_Numpy(...)                    # Or FA_Standard() / FA_Vectorised()
predictions = FA_object.fit(...)             # Run the EM algorithm on the generated data.
```

Config Files
```python
cfg = {
    'METHOD': 'Standard',     # Which implementation to use
    'PLOT_GRAPHS': True,      # Plot graphs?
    'NUM_FEATURES' : 50,      # Number of features (p)
    'NUM_FACTORS' : 12,       # Number of factors (k)
    'NUM_SAMPLES' : 20,       # Number of sampled points (n)
    'NUM_ITERATIONS' : 1000,  # Iterations of EM
    'LOG_FREQ' : 1,           # Frequency of logs 
    'RANDOM_SEED' : 1,        # Seed for reproducibility
}

```

## Directory Structure
```
├── generate_plots.py                   # Generate the Plots shown above. [*]
├── configs.py                          # Stores config files             
├── example.py                          # Example of API usage
├── pytorch_factor_analysis             # Main Package
│   ├── __init__.py
│   ├── Factor_Analysis_Numpy.py            # Standard implementation  (Numpy)
│   ├── Factor_Analysis_Standard.py         # Standard implementation  (PyTorch)
│   ├── Factor_Analysis_Vectorised.py       # Vectorised implementation  (PyTorch)
│   └── utils.py                            # Utils 
├── figures                             # Figures from `generate_plots.py`
├── generate_plots.log                  # Output log from `generate_plots.py`
├── setup_environment.sh                # Script to create Conda environment
├── setup.py                            # Script to set up package
├── README.md 
└── LICENSE
```

## Known issues
Although around 100x faster than the `Numpy` and `Standard` implementations the `Vectorised` implementation often breaks for some test cases due to the inability to invert Singular matrices. These cases throw an error similar to: `RuntimeError: inverse_cpu: U(x,x) is zero, singular U.`


## References


## License