.PHONY: up
up:
    docker build -t magneton:latest .
run:
   docker run -ti magneton:latest /bin/bash