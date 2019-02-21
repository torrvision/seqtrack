#!/bin/bash

# Print warning if CUDA_VISIBLE_DEVICES is not set to empty string.
# (Running tests on GPU is non-deterministic.)
if [ -z "${CUDA_VISIBLE_DEVICES+qwerty}" ]; then
    echo "WARNING: CUDA_VISIBLE_DEVICES is not set"
elif [ ! -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "WARNING: CUDA_VISIBLE_DEVICES is non-empty: \"${CUDA_VISIBLE_DEVICES}\""
fi

# nosetests -v --with-doctest --nologcapture seqtrack
pytest -v --doctest-modules python/seqtrack/
