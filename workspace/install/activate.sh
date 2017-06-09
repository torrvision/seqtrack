if [ -d env ] ; then
    source env/bin/activate
fi

export PYTHONPATH="$(readlink -f ./repo/):$PYTHONPATH"
