#!/bin/sh
set -e

./extract > myfeatures.json
#./fit --l1 1.0 < myfeatures.json > weights.json
./fit --l2 1.0 < myfeatures.json > weights.json
./score -w weights.json < myfeatures.json > output.txt
./evaluate < output.txt
