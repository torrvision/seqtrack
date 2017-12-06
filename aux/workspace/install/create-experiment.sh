#!/bin/bash

commit=$(cat commit.txt)
remote=$(cat remote.txt)

if ! git clone "$remote" repo ; then
    echo 'cannot clone repository'
    exit 1
fi

if ! ( cd repo && git checkout "$commit" ) ; then
    echo 'cannot checkout commit (make sure it is pushed)'
    exit 1
fi

# if ! virtualenv env ; then
#     echo 'could not create virtual environment'
#     exit 1
# fi

# # Optional local virtual environment.
# if [ -f requirements.txt ] ; then
#     touch pip.conf
#     if ! cp pip.conf env/pip.conf ; then
#         echo 'could copy pip configuration'
#         exit 1
#     fi
#     if ! env/bin/pip install --upgrade pip ; then
#         echo 'could not upgrade pip'
#         exit 1
#     fi
#     if ! env/bin/pip install -r requirements.txt ; then
#         echo 'could not install pip requirements'
#         exit 1
#     fi
# fi

echo 'create workspace successful'
