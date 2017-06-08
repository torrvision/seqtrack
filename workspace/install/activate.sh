if [ -d env ] ; then
    source env/bin/activate
fi

export PYTHONPATH="$(realpath ./repo/):$PYTHONPATH"
