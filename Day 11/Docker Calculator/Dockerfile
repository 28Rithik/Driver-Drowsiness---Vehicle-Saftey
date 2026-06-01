# Use the official Node.js image as the base image for the Docker container from the Docker Hub
FROM node:20
# Set the working directory in the container to /calculator
WORKDIR /calculator
# Copy the 'calculator.html' and 'server.js' files from the local machine to the working directory in the container
COPY calculator.html .
COPY server.js .
# Expose port 5000 to allow external access to the server running inside the container
EXPOSE 5000 
# Set the default command to run the server using Node.js when the container starts
CMD [ "node","server.js" ]