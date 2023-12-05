#!/bin/bash

# Downsample images
mkdir -p train_data
counter=0
for i in {0001..0300}; do
    num=$(echo $i | sed 's/^0*//')
    if [ $(($num % 10)) -eq 0 ]; then
        echo $i
        convert -resize 50% -flatten +matte cylinder/fine/vorticity/image.$i.png train_data/$counter.png
        ((counter+=1))
    fi
done

# Create captions
python make_metadata.py train_data/ cylinder
