#!/bin/bash

commit=$(cat commit.txt)
remote=$(cat remote.txt)

if ! git clone "$remote" repo ; then
	echo 'cannot clone repository'
	exit 1
fi

if ! ( cd repo && git checkout "$commit" ) ; then
	echo 'cannot checkout commit'
	exit 1
fi

if ! virtualenv env ; then
	echo 'could not create virtual environment'
	exit 1
fi

if ! env/bin/pip install -r requirements.txt ; then
    echo 'could not install pip requirements'
    exit 1
fi

echo 'create workspace successful'
