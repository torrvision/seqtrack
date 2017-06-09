#!/bin/bash

if [[ $# -ne 2 ]] ; then
    echo "usage: $0 experiments/ name"
    exit 1
fi

root="$1"
name="$2"

# Move to parent directory of script.
script_file="$( readlink -f "${BASH_SOURCE[0]}" )"
script_dir="$( dirname "$script_file" )"
src="$( dirname "$script_dir" )"
src_project="$( dirname "$src" )"

date=$(date +%Y-%m-%d)
fullname="$date-$name"

dir="$root/$fullname"
echo "workspace: $dir"

if ! mkdir "$dir" ; then
    echo 'cannot create directory'
    exit 1
fi

if ! cp $src/workspace/install/* "$dir/" ; then
    echo 'cannot copy workspace scripts'
    exit 1
fi
if ! mkdir "$dir/workspace" ; then
    echo 'cannot create workspace directory'
    exit 1
fi
if ! touch "$dir/workspace/run.sh" ; then
    echo 'cannot touch run.sh'
    exit 1
fi
if ! (cd $src && git log -1 --format="%H") >"$dir/commit.txt" ; then
    echo 'cannot get git commit'
    exit 1
fi
if ! (cd $src && git config --get remote.local.url) >"$dir/remote.txt" ; then
    echo 'cannot get git remote url'
    exit 1
fi
# Optional local virtual environment.
virtualenv="$src_project/env"
if [ -d $virtualenv ] ; then
    if ! echo "$virtualenv" >"$dir/virtualenv.txt" ; then
        echo 'cannot write virtualenv to file'
    fi
    if ! ($virtualenv/bin/pip freeze) >"$dir/requirements.txt" ; then
        echo 'cannot get pip requirements'
        exit 1
    fi
    # touch "$src/env/pip.conf"
    # if ! cp "$src/env/pip.conf" "$dir/pip.conf" ; then
    #     echo 'cannot copy pip configuration'
    #     exit 1
    # fi
fi

( cd $dir &&  ./create-experiment.sh )

# TODO: Come up with a better place to put aux/ files?
if ! rsync -a "$src/aux/" "$dir/repo/aux/" ; then
    echo 'cannot copy auxiliary files'
    exit 1
fi

echo "workspace: $dir"
