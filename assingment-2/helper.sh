#!/usr/bin/bash

for dir in image-{1..5}
do
    rm -rf results/$dir/image-patches/*.jpg
done;