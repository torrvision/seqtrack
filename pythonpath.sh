repo="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$repo:$PYTHONPATH"
export PYTHONPATH="$repo/trackdat/python:$PYTHONPATH"
export PYTHONPATH="$repo/slurmproc:$PYTHONPATH"
