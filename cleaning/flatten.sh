#!/bin/bash

# takes 

target="${1:-.}"

find "$target" -mindepth 2 -type f | while read -r file; do
    filename=$(basename "$file")
    dest="$target/$filename"

    if [ -e "$dest" ]; then
        base="${filename%.*}"
        ext="${filename##*.}"
        [ "$base" = "$filename" ] && ext="" || ext=".$ext"
        counter=1
        while [ -e "$target/${base}_${counter}${ext}" ]; do
            ((counter++))
        done
        dest="$target/${base}_${counter}${ext}"
    fi

    mv "$file" "$dest"
done

find "$target" -mindepth 1 -type d -empty -delete
