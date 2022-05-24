# # This file contains information on how you should structure your docker image for submission.

# # Always use alpine version to minimize docker image size. If alpine version 
# # is not available, use the smallest base image available.
# FROM python:3-alpine
# # from smizy/scikit-learn:latest

# # This will be the base directory where our project will live. 
# # The name can be anything but for now, let's name it client since 
# # the bot is a single client in our game.
# WORKDIR client

# # ADD command adds the file or folder to the destination. 
# # Since the working directory is `./client`, it copies the file inside `./client`.
# RUN mkdir ./src/
# ADD ./src/* ./src/
# ADD ./requirements.txt .
# Add ./filename.joblib .

# # RUN commands are run when the docker image is built. 
# # Prefer to use RUN commands to install packages, build packages 
# # and stuff that needs to be done only once.
# # RUN command runs the command in command line
# # build-base adds gcc and other tools required to build sanic
# RUN apk add build-base
# RUN pip3 install -r requirements.txt


# # EXPOSE opens up the port to communication outside the container.
# # WE ASSUME THAT YOUR SERVER WILL RUN ON THIS PORT. 
# # DO NOT CHANGE THIS.
# EXPOSE 7000

# # CMD runs the specified command on docker image startup.
# # Note that we are inside the working directory `./client` so, 
# # `python app.py` is run inside the `./client` directory.
# CMD DEBUG=0 python3 src/app.py



# # Use an official Python runtime as a parent image
FROM python:3-slim
# # FROM continuumio/miniconda3:latest
# FROM pypy:latest

# # Set the working directory to /app
WORKDIR /app


RUN mkdir ./src
ADD ./src/* ./src/
ADD ./requirements.txt .

# # Copy the current directory contents into the container at /app
# ADD ./src/app.py .
# # ADD ./src/bot.py .
# # ADD ./src/card.py .
# # ADD ./src/ismcts.py .

Add ./filename.joblib .


# COPY ./src/* .

# # Install any needed packages specified in requirements.txt
# # RUN conda install -c conda-forge pypy
# # RUN conda config --set channel_priority strict
# # RUN conda install -c conda-forge pypy
# # RUN conda create -n pypy pypy
# # RUN conda activate pypy
# # SHELL ["conda", "run", "-n", "pypy", "/bin/bash", "-c"]
# # RUN conda activate pypy
RUN pip install  -r requirements.txt
# RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# # Define environment variable
# # ENV NAME World

EXPOSE 7000

# # Run app.py when the container launches
# # CMD ["python", "app.py"]
CMD DEBUG=0 python3 src/app.py
