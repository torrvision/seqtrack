repo="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$repo/python:$PYTHONPATH"
export PYTHONPATH="$repo/submodules/trackdat/python:$PYTHONPATH"
export PYTHONPATH="$repo/submodules/slurmproc:$PYTHONPATH"
