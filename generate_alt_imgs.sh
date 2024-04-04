#!/bin/bash
is=(25, 50, 75, 100)

for i in is
do
   echo "Running iteration $i"
   python generate_imgs.py
done