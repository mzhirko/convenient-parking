# Dockerized Version of Convenient Parking

It turned out to be hard to install and run all the dependencies MCNN needs on some OS and OS distros, so there is a possible solution for an attempt to *_Dockerize_* this code.

See the steps of *_Docker installation_* following the link.

After that u're gonna need to know & use following commands, so there's a *Cheatsheet*

1. Run that command to build the docker image
```Terminal
docker build -t parking .
```
2. To get the running app container, type:
```Terminal
docker run parking
```
Unhopefully i wasn't able to test it yet 'cause of the downloading weights file size :^) it takes way too long to complete.
And i think we're gonna need to know how to share GPU to the working Docker Container. Cuz that's not done yet :)))

3. Heading to the next useful commands:

Deletes Docker image:
```Terminal
docker images -a
docker rmi Image_Name
```
Deletes all Docker images:
```Terminal
docker rmi $(docker images -a -q)
```
Deletes Docker container:
```Terminal
docker ps -a
docker rm ID_or_Name
```
Deletes all Docker containers:
```Terminal
docker ps -a | xargs docker rm
```

*That's about all* :octocat: :two_hearts: