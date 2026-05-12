# Notice

This package combines local project code with converted model artifacts.

## Driver and Runtime

The CC OpenCL driver and Python runtime are included for local research and engineering use. Confirm the intended license before publishing this package outside the local project.

## Model Weights

The `.ccq4` files are converted from a Gemma-family model checkpoint. The upstream Gemma license, model access terms and acceptable use terms continue to apply to these derived weights.

Do not upload or redistribute the converted `.ccq4` weights unless you have confirmed that the upstream terms permit that distribution.

## OpenCL

OpenCL headers and import helper files are included to support local Windows builds of `CC_OpenCl.dll`. The installed AMD OpenCL runtime is still required at execution time.

