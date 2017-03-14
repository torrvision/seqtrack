#!/bin/bash

if [[ $# -ne 2 ]] ; then
	echo "usage: $0 experiments/ name"
	exit 1
fi

root="$1"
name="$2"

# Move to parent directory of script.
scriptdir="$( dirname "${BASH_SOURCE[0]}" )"
src="$(dirname "$scriptdir")"

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
if ! (cd $src && git log -1 --format="%H") >"$dir/commit.txt" ; then
	echo 'cannot get git commit'
	exit 1
fi
if ! (cd $src && git config --get remote.origin.url) >"$dir/remote.txt" ; then
	echo 'cannot get git remote url'
	exit 1
fi
if ! (cd $src && env/bin/pip freeze) >"$dir/requirements.txt" ; then
	echo 'cannot get pip requirements'
	exit 1
fi

( cd $dir &&  ./create-experiment.sh )

echo "workspace: $dir"
