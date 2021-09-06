# CMB_GP_clustering_analysis
Clustering analysis for foregrounds of CMB

## Docker stuff
- To build a docker image from the docker-compose use the following command in the terminal in the folder where the docker-compose.yaml is located

  ` sudo -E docker-compose up --build `

- Once the image is built it can saved a tar file, ready to be shipped by running the following command. This will create a .tar.gz file (with the name of the docker image replaceing myimage).

  ` sudo docker save myimage:latest | gzip > myimage_latest.tar.gz `

- If you just obtained a .tar.gz file and you wish to just run the docker image, enter the following command to load in the .tar file.

  ` sudo docker load < myimage_latest.tar.gz `
  
- Once the tar file is loaded, run the following command to run the docker image

  ` sudo docker-compose up `
