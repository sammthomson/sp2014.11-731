#!/bin/sh
./extract > myfeatures.json
./fit --l2 1.0 < myfeatures.json > weights.json
./score -w weights.json < myfeatures.json > output.txt
./evaluate < output.txt
