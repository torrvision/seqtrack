#!/bin/bash

root=$1 # path/to/ILSVRC2015
if [ "$#" -ne 1 ]; then
    echo "usage: $0 data/ILSVRC"
    exit 1
fi

src=$root/Data
dst=$root/Data_640_360

if [ ! -d "$src" ]; then
    echo "directory does not exist: $src"
    exit 1
fi

mkdir -p $dst

for subset in train val
do
    # Create directory structure.
    (cd $src && find VID/$subset -type d) | \
        xargs -I{} mkdir -p $dst/{}
    # Resize images.
    (cd $src && find VID/$subset -type f) | \
        xargs -n 1 -P 16 -I{} convert $src/{} -resize '640x360>' $dst/{}
done
