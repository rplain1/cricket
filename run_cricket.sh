#!/bin/bash

git clone https://github.com/rplain1/cricket.git

cd cricket || exit

docker build -t cricket .

docker run --rm cricket
