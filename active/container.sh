#!/bin/bash

# Define variables
IMAGE_NAME="leam-us"
CONTAINER_NAME="leam-us-container"
PORT=8888  # Change if needed

echo "Starting containerization process..."

# Step 1: Build the Docker image
echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

# Step 2: Check if a container with the same name is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Step 3: Run the container
echo "Running container: $CONTAINER_NAME"
docker run -d --name $CONTAINER_NAME -p $PORT:$PORT $IMAGE_NAME

echo "Container is running. Access Jupyter Notebook at: http://localhost:$PORT"
