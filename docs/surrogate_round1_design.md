# Surrogate Round-1 Design

## Input Space

- gauge-fixed pairing raw parameter vector
- `barrier_z`
- `gamma`

The round-1 normal-state sector is fixed to `base_normal_state_params()`.

## Output Space

- broadened conductance on a fixed bias grid
- default grid: `[-40, 40] meV` with `601` points

## Model Class

- first version uses a residual MLP
- no transformer, diffusion, or flow-based model is introduced in round 1

## Training Requirements

- train/val/test split
- fixed random seed
- early stopping
- checkpoint save
- config export
- training log save

## Inverse Contract

The surrogate accelerates candidate search only. Final candidate ranking is
re-checked with the real physics forward workflow.
