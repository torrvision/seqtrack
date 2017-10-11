#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 src/ dst/ size"
    echo
    echo "Examples for size:"
    echo "    '241x241!'  -- stretch to 241x241"
    echo "    '640x360>'  -- shrink to fit 640x360"
    echo "    '360x360^>' -- shrink to fill 360x360"
    exit 1
fi
src="$1"
dst="$2"
size="$3"

if [ ! -d "$src" ]; then
    echo "directory does not exist: $src"
    exit 1
fi

mkdir -p "$dst"

# Create directory structure.
(cd "$src" && find . -type d) | \
    xargs -I{} mkdir -p "$dst/{}"
# Resize images.
(cd "$src" && find . -type f -iname '*.gif' -o -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg') | \
    xargs -n 1 -P 16 -I{} convert "$src/{}" -resize "$size" "$dst/{}"
