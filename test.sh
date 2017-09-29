#!/bin/bash

# Print warning if CUDA_VISIBLE_DEVICES is not set to empty string.
if [ -z "${CUDA_VISIBLE_DEVICES+qwerty}" ]; then
    echo "WARNING: CUDA_VISIBLE_DEVICES is not set"
elif [ ! -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "WARNING: CUDA_VISIBLE_DEVICES is non-empty: \"${CUDA_VISIBLE_DEVICES}\""
fi

python -m unittest discover .
