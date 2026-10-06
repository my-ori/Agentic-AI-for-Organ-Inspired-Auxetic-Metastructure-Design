# Patch-design workflow guidance

## Scope

These instructions apply to this repository and every directory below it.
`workflow.txt` is the scientific specification. Preserve its production parameters unless the user explicitly approves a change.

## Runtime environments

- Run GPU diffusion, epsilon processing, plotting, selection, extraction, and MAE calculation in WSL Ubuntu with:
  `/home/yingbchen/miniconda3/envs/cnn_gpu/bin/python`
- Run mesh conversion with:
  `/home/yingbchen/miniconda3/envs/sfepy-env2/bin/python`
- Run finite-element solves with:
  `/home/yingbchen/miniconda3/envs/sfepy-env2/bin/sfepy-run`
  The finite-element solves can run in parallel. 
- Production diffusion must use CUDA. Do not silently fall back to CPU.
- Use absolute paths when commands cross case or environment boundaries.

## Production safeguards

- Process GPU stages serially. Do not run two diffusion samplers concurrently on the same GPU.
- Treat existing user data as authoritative. Skip validated completed outputs; do not delete or overwrite them merely to restart a workflow.
- Resume an incomplete epsilon sweep with `--skip_existing`.
- Keep case outputs inside their corresponding directory under `organ_inspired_cases`.
- Capture stage logs and verify required artifacts before advancing.

## Workflow interpretation

- Generate ten logarithmic epsilon values with `epsilon_dividing.py`; this excludes the range start and includes `1.0`, matching the current script.
- Each selected design must have its own directory:
  `5samples/sample_01` through `5samples/sample_05`.
- Each FE solve runs from that sample's `fe` directory because `patch_tension.py` reads `design_000_tile3x3.vtk` from the current working directory.
- Preserve `out_bitmap2d/reaction_right_grip.csv` and copy it to the sample's `fe/reaction_right_grip.csv` for the documented MAE step.

## Validation and reporting

- A sweep is complete only when it reports no failed or missing runs and its merged CSV/PT artifacts exist.
- At handoff, report completed stages, failures, selected source files, FE status, and stress/Poisson-ratio MAEs for every case.
