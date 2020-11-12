# PyTorch Factor Analysis

![Factor Analysis](https://github.com/adiah80/PyTorch-Factor-Analysis/blob/master/figures/Factor_analysis.png)

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
import pytorch_factor_analysis as pfa        # Do the imports.

device = torch.device(...)                   # Set the device for PyTorch.
cfg = {...}                                  # Set the run config file.

data = pfa.generate_sample_data(...)         # Sample Data for the run.
FA_object = pfa.FA_Numpy(...)                # Or FA_Standard() / FA_Vectorised()
predictions = FA_object.train_EM(...)        # Run the EM algorithm on the generated data.
pfa.plot(...)                                # Plot the training graphs
```

Config Files

Config files for `example.py` should be as shown below. Some sample config files can be found in the `config` folder.
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

## Training Curves

![Lambda Error](https://github.com/adiah80/PyTorch-Factor-Analysis/blob/master/figures/Lambda_error.png)

![Psi Error](https://github.com/adiah80/PyTorch-Factor-Analysis/blob/master/figures/Psi_error.png)

## Known issues
Although around 100x faster than the `Numpy` and `Standard` implementations the `Vectorised` implementation often breaks for some test cases due to the inability to invert Singular matrices. 

These cases throw an error similar to: `RuntimeError: inverse_cpu: U(x,x) is zero, singular U.`

## References

- The EM Algorithm for Mixtures of Factor Analyzers. Zoubin Ghahramani, Georey E. Hinton. \[ [PDF](http://www.cs.toronto.edu/~fritz/absps/tr-96-1.pdf) ]
- Lectures on Factor Analysis (CMU). Leibny Paola García Perera.  \[ [PDF](https://www.cs.cmu.edu/~pmuthuku/mlsp_page/lectures/slides/JFA_presentation_final.pdf) ]
- Stanford CS229 - EM Algorithm & Factor Analysis. Andrew Ng.  \[ [PDF - EM](http://cs229.stanford.edu/notes2020spring/cs229-notes8.pdf
) ]  \[ [PDF - Factor Analysis](http://cs229.stanford.edu/notes2020spring/cs229-notes9.pdf) ] \[ [Video](https://www.youtube.com/watch?v=tw6cmL5STuY
) ]

## License
MIT License
