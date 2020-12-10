# set base image (host OS)
FROM python:3.6


# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY ./source/requirements.txt .

# update and install libgl to avoid error in opengl
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN apt-get install libgtk2.0-dev -y
# install + upgrade pip
RUN python -m pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY source/ .

# command to run on container start
CMD [ "python", "./parking.py" ] 
