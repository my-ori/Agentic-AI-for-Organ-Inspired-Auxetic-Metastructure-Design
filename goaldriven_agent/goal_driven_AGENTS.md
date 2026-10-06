# Background

This folder contains several completed organ-inspired patch design cases generated using the baseline workflow described in workflow.txt.
To further enhance the design, we will undertake the following tasks.


# Mission

For each organ-inspired case, use the current five final designs as the baseline.
The stress MAE and Poisson-ratio MAE values for all five designs are already available in their respective folders. Use these values to construct the corresponding two-objective Pareto frontier.
Then, generate a new, physically valid design (x^*) that breaks through the existing Pareto frontier.
A successful design (x^*) must achieve at least a 30% reduction in both stress MAE and Poisson-ratio MAE relative to at least one design (x_i) on the existing Pareto frontier.
Specifically, there must exist an original Pareto-optimal design (x_i) such that
MAE_stress(x^*) <= 0.70 MAE_stress(x_i) and MAE_Poisson(x^*) <= 0.70 MAE_Poisson(x_i).
Both conditions must be satisfied simultaneously with respect to the same Pareto-optimal design (x_i).


# Immutable scientific contract

Do not modify:
- the target stress-strain or Poisson's-ratio curves;
- the material properties;
- the FE boundary conditions;
- the reference MAE definitions;
- the reference FE procedure.

# Acceptance conditions

A successful design must:
1. satisfy connectivity and periodicity;
2. pass independent FE verification;
3. the calculation of MAE should be consistent with the code 'compute_mae.py'.

# Autonomy

Choose and implement any computational strategy you judge scientifically appropriate. 
You may create new code, train additional models, run simulations, and revise your strategy. 
The new design may be generated from scratch or obtained by refining any original Pareto-optimal design (x_i).
No optimization method is prescribed.

# Required procedure

1. Log every candidate and evaluation.
2. Stop at completing one successful design.
6. Submit the final report.

## Runtime environments

- Run GPU diffusion, epsilon processing, plotting, selection, extraction, and
  MAE calculation in WSL Ubuntu with:
  `/home/yingbchen/miniconda3/envs/cnn_gpu/bin/python`
- Run mesh conversion with:
  `/home/yingbchen/miniconda3/envs/sfepy-env2/bin/python`
- Run finite-element solves with:
  `/home/yingbchen/miniconda3/envs/sfepy-env2/bin/sfepy-run`
  The finite-element solves can run in parallel. 