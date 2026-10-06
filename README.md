# Agentic AI for Organ-Inspired Auxetic Metastructure Design

This repository contains the computational workflows, agent instructions, and supporting code associated with the study:

**Agentic AI for Organ-Inspired Auxetic Metastructure Design**

The study investigates the use of a general-purpose coding agent for the computational design of organ-inspired auxetic metastructure patches. Two agent configurations are considered:

1. **Workflow agent** – executes a prescribed inverse-design workflow while preserving its predefined scientific procedure and parameters.
2. **Goal-driven agent** – receives a quantitative design goal and fixed scientific constraints but is free to formulate and execute its own computational strategy.

A human-executed implementation of the prescribed workflow is also included as the reference configuration.

---

## Repository Structure

```text
.
├── goaldriven_agent/
│   └── goal_driven_AGENTS.md
│
├── human_researcher/
│   ├── calculate_physical_mae.py
│   ├── compute_mae.py
│   ├── config.json
│   ├── diffusion/
│   │   ├── ddpm.py
│   │   ├── ema.py
│   │   ├── image_io.py
│   │   ├── unet.py
│   │   └── utils.py
│   ├── epsilon_dividing.py
│   ├── extract_epsilon_candidates.py
│   ├── extract_tile_and_vtk_from_guided_pt.py
│   ├── guided_sample_diffusion_epsilon_constraint.py
│   ├── guided_sample_diffusion_flexible_target.py
│   ├── models/
│   │   └── cnn_surrogate.py
│   ├── organ_inspired_cases/
│   │   ├── human_bladder/
│   │   │   └── y_target.json
│   │   ├── human_heart/
│   │   │   └── y_target.json
│   │   ├── porcine_thoracic_aorta/
│   │   │   └── y_target.json
│   │   ├── rat_lung/
│   │   │   └── y_target.json
│   │   └── rat_stomach/
│   │       └── y_target.json
│   ├── patch_npy_to_vtk_inp_stage2.py
│   ├── patch_tension.py
│   ├── plot_pareto_scatter_physical_mae.py
│   ├── select_samples_physical_mae.py
│   ├── sweep_epsilon_runs_and_merge.py
│   └── workflow.txt
│
└── workflow_agent/
    ├── workflow_agent_AGENTS.md
    └── workflow.txt
```

---

## Experimental Configurations

### Human Researcher

The `human_researcher/` directory contains the computational tools used to execute the prescribed inverse-design workflow.

The complete workflow sequence and production parameters are documented in:

```text
human_researcher/workflow.txt
```

### Workflow Agent

The `workflow_agent/` directory contains the instructions provided to the workflow-agent system:

```text
workflow_agent/workflow_agent_AGENTS.md
workflow_agent/workflow.txt
```

To reproduce the workflow-agent experiment using Codex, copy the computational files and directories from `human_researcher/` into `workflow_agent/`. This provides the agent with the same computational tools, model definitions, target-response files, and configuration used by the human researcher, together with the workflow and agent-specific instructions provided in this directory.

### Goal-Driven Agent

The `goaldriven_agent/` directory contains the instructions provided to the goal-driven agent:

```text
goaldriven_agent/goal_driven_AGENTS.md
```

To reproduce the goal-driven-agent experiment using Codex, copy the contents of the completed `workflow_agent/` directory into `goaldriven_agent/`. This provides the goal-driven agent with the completed baseline designs and computational tools from the workflow-agent experiment, together with the goal-driven instructions provided in this directory.

---

## Organ-Inspired Design Cases

Five organ-inspired mechanical-response cases are considered:

- Human heart
- Rat lung
- Human bladder
- Porcine thoracic aorta
- Rat stomach

The prescribed target responses for each case are stored in:

```text
human_researcher/organ_inspired_cases/human_heart/y_target.json
human_researcher/organ_inspired_cases/rat_lung/y_target.json
human_researcher/organ_inspired_cases/human_bladder/y_target.json
human_researcher/organ_inspired_cases/porcine_thoracic_aorta/y_target.json
human_researcher/organ_inspired_cases/rat_stomach/y_target.json
```

Each case is represented by prescribed stress–strain and strain-dependent effective Poisson's-ratio responses used as the mechanical targets for inverse design.

Details regarding the experimental sources, preprocessing, and construction of these organ-inspired targets are provided in the accompanying manuscript.

---

## Main Computational Scripts

### Diffusion Sampling

```text
guided_sample_diffusion_flexible_target.py
guided_sample_diffusion_epsilon_constraint.py
sweep_epsilon_runs_and_merge.py
```

These scripts perform diffusion-based candidate generation and epsilon-constraint guided sampling.

### Epsilon Calibration

```text
extract_epsilon_candidates.py
epsilon_dividing.py
```

These scripts determine the epsilon-constraint range and generate the epsilon values used for the guided sampling sweep.

### Surrogate Evaluation and Pareto Analysis

```text
calculate_physical_mae.py
plot_pareto_scatter_physical_mae.py
select_samples_physical_mae.py
```

These scripts calculate surrogate-based response errors, construct the Pareto representation, and select representative designs for FE verification.

### Finite-Element Analysis

```text
extract_tile_and_vtk_from_guided_pt.py
patch_npy_to_vtk_inp_stage2.py
patch_tension.py
compute_mae.py
```

These scripts convert selected geometries into FE models, perform FE simulations, and calculate FE-based response MAEs.

---

## Pretrained Models

The prescribed inverse-design workflow requires pretrained CNN and DDPM checkpoints.

The principal model files used in the study are:

```text
best.pt
ddpm_epoch_2950.pt
config.json
```

Because the pretrained CNN and DDPM checkpoint files are too large to host directly in this GitHub repository, they are provided separately.

### Download Pretrained Checkpoints

**CNN checkpoint (`best.pt`):**  
[https://drive.google.com/file/d/15DqcAVtY4yEr0D6PH5NUIecrM1MuURIi/view?usp=sharing]

**DDPM checkpoint (`ddpm_epoch_2950.pt`):**  
[https://drive.google.com/file/d/1WZmzYnS0JmEFVk0fnFlgR-PC0ZGjd-iP/view?usp=sharing]

The CNN configuration file:

```text
config.json
```

is included directly in this repository.

After downloading the checkpoints, place them in the working directory from which the prescribed workflow is executed, or update the checkpoint paths supplied to the corresponding Python scripts.

---

## Computational Environment

The original computations were performed under Linux/WSL using separate Python environments for machine-learning operations and FE simulation.

### Machine-Learning Environment

The principal dependencies include:

```text
Python
PyTorch
NumPy
SciPy
pandas
Matplotlib
```

CUDA-capable GPU acceleration is required for the production diffusion-sampling stages used in the study.

### Finite-Element Environment

A separate environment containing SfePy was used for:

- FE mesh/model preparation; and
- nonlinear FE simulations.

The principal FE package is:

```text
SfePy
```

---

## Related Work

The diffusion-based inverse-design framework used in this study builds on our previous work:

> Y. Chen, B. Graham, X. Mu, and S. Xiao,  
> **"Diffusion-Based Inverse Design of Organ-Inspired Auxetic Metastructure Patches."**  
> *Frontiers in Materials*, 2026.

The code and data associated with that study are available at:

https://github.com/my-ori/Diffusion-Based-Inverse-Design-of-Organ-Inspired-Auxetic-Metastructure-Patches

---

## License

This repository is released under the **MIT License**.

Department of Mechanical Engineering  
University of Iowa
