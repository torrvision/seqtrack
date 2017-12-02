#!/bin/bash

if [[ "$#" -ne 2 ]] ; then
    echo "usage: "$0" experiments/ name"
    exit 1
fi

experiments_root="$1"
name="$2"

# Move to parent directory of script.
script_file="$( readlink -f "${BASH_SOURCE[0]}" )"
script_dir="$( dirname "$script_file" )"
# root/aux/workspace/init-experiment.sh
orig_repo_dir="$( dirname "$( dirname "$script_dir" )" )"

date="$(date +%Y-%m-%d)"
fullname="$date-$name"

dir="$experiments_root/$fullname"
echo "workspace: $dir"

if ! mkdir "$dir" ; then
    echo 'cannot create directory'
    exit 1
fi

if ! cp "$script_dir"/install/* "$dir/" ; then
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
if ! (cd "$script_dir" && git log -1 --format="%H") >"$dir/commit.txt" ; then
    echo 'cannot get git commit'
    exit 1
fi
if ! (cd "$script_dir" && git config --get remote.local.url) >"$dir/remote.txt" ; then
    echo 'cannot get git remote url'
    exit 1
fi

# # Optional local virtual environment.
# if [ -d "$orig_repo_dir/env" ] ; then
#     if ! "$orig_repo_dir/env/bin/pip" freeze >"$dir/requirements.txt" ; then
#         echo 'cannot get pip requirements'
#         exit 1
#     fi
#     touch "$orig_repo_dir/env/pip.conf"
#     if ! cp "$orig_repo_dir/env/pip.conf" "$dir/pip.conf" ; then
#         echo 'cannot copy pip configuration'
#         exit 1
#     fi
# fi

( cd "$dir" &&  ./create-experiment.sh )

# TODO: Come up with a better place to put aux/ files? Maybe with data?
aux_data_dir=seqtrack/aux
if ! rsync -a "$orig_repo_dir/$aux_data_dir/" "$dir/repo/$aux_data_dir/" ; then
    echo 'cannot copy auxiliary files'
    exit 1
fi

echo "workspace: $dir"
