# CMB_GP_clustering_analysis
Clustering analysis for foregrounds of CMB

## Introduction

In this repo we provide a program to find regions of the microwave sky that are similar. More rigoursouly, we use a clustering algorithm to take sky maps produced by simulating the CMB signal, as well as the foreground and noise signals, to find clusters with similar properties. This is done directly on the map values in each pixel, as opposed to using spectral indicies or other parameteric variables that are often used for parametrising the sky. Thus, our technique is the most data driven method one can take. 

## Docker stuff
- To build a docker image from the docker-compose use the following command in the terminal in the folder where the docker-compose.yaml is located

  ` sudo -E docker-compose up --build `

- Once the image is built it can saved a tar file, ready to be shipped by running the following command. This will create a .tar.gz file (with the name of the docker image replaceing myimage).

  ` sudo docker save myimage:latest | gzip > myimage_latest.tar.gz `

- If you just obtained a .tar.gz file and you wish to just run the docker image, enter the following command to load in the .tar file.

  ` sudo docker load < myimage_latest.tar.gz `
  
- Once the tar file is loaded, run the following command to run the docker image

  ` sudo docker-compose up `
