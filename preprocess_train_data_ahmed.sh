#!/bin/bash

# Downsample images
mkdir -p train_data
counter=0
for i in {0001..0125}; do
    num=$(echo $i | sed 's/^0*//')
    if [ $(($num % 3)) -eq 0 ]; then
        echo $i
        convert -resize 100% -flatten +matte ahmed/des/image.$i.png train_data/$counter.png
        ((counter+=1))
    fi
done

# Create captions
python make_metadata.py train_data/
