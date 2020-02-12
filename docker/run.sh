#!/bin/bash

docker build --file docker/Dockerfile \
	     --tag attre:dev \
	     $PWD

docker run -d \
	   -v "${PWD}:/app" \
	   --runtime=nvidia \
	   -e NVIDIA_VISIBLE_DEVICES=0 \
	   --name attre-docker \
	   attre:dev
