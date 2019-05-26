# SPEMD
Successful Parameter Enumerator for MD simulation


## Requirements
1. Python >= 2.7
2. [COMBO](https://github.com/tsudalab/combo): A Bayesian optimization library 
3. numpy
4. scikit-learn 
5. scipy

## Usage
- Preparation
  - Please list all parmeter candidates as a CSV file. (See example/parameter_list(Newtonian).csv or example/parameter_list(Langevin).csv)
  - Please call your MD simulation in the simulation function in simulator.py and return its success rate.

## Enumeration examples of F1-motor simulation based on CG-MD
- Newtonian dynamics version.
  - python parameter_enumerator.py 21,12 100 example/parameter_list\(Newtonian\).csv 1.0 --method BOUS --test Newtonian
  - BOUS based search with the success threshold of 1.0 using 100 samplings
 
- Langevin dynamics version.
  - python parameter_enumerator.py 30,9,5 400 example/parameter_list\(Langevin\).csv 0.8 --method BOUS --test Langevin
  - BOUS based search with the success threshold of 0.8 using 400 samplings
