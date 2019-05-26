# SPEMD
Successful Parameter Enumerator for MD simulation


## Requirements
1. Python >= 2.7
2. [COMBO](https://github.com/tsudalab/combo): A Bayesian optimization library 
3. numpy >= 1.15.0
4. scikit-learn >= 0.19.1
5. scipy >=  1.1.0

## Installation
Download or clone the github repository, e.g. git clone https://github.com/tsudalab/SPEMD

## Usage
- Preparation
  - Please list all parmeter candidates as a CSV file. (See example/parameter_list(Newtonian).csv or example/parameter_list(Langevin).csv)
  - Please call your MD simulation in the simulation function in simulator.py and return its success rate.

- Sucessful parameter enumeration based on machine learning algorithms. (See the commands of F1-motor examples in the following.)
  - `python parameter_enumerator.py [Comma separated numbers of candidate for each variable] [Number of sampling iterations] [Directory of the parameter candidate file] [Successful threshold] --method [Search method]`
  - Available search methods: BOUS (Combination of BO and US), BO (Bayesian Optimization), US (Uncertainty Sampling), RS (Random Sampling)

## Enumeration examples of F1-motor simulations based on CG-MD
- Newtonian dynamics version.
  - `python parameter_enumerator.py 21,12 100 example/parameter_list\(Newtonian\).csv 1.0 --method BOUS --test Newtonian`
  - BOUS based search with the success threshold of 1.0 using 100 samplings
 
- Langevin dynamics version.
  - `python parameter_enumerator.py 30,9,5 400 example/parameter_list\(Langevin\).csv 0.8 --method BOUS --test Langevin`
  - US based search with the success threshold of 0.8 using 400 samplings
