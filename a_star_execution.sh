#!/bin/bash

for i in {1..25}
do
 python src/a_star_coverage.py --num_drones 2 --seed $i
 python src/a_star_coverage.py --num_drones 4 --seed $i
done