#!/bin/bash

# Clone the repository
git clone https://github.com/rplain1/cricket.git

# Change directory to the cloned repo
cd cricket || exit

# Build the Docker image
docker build -t cricket .

# Run the Docker container interactively
docker run -it cricket
