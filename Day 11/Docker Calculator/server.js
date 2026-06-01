// Importing the 'http' module to create an HTTP server
const http = require('http');
// Importing the 'fs' module to read files and 'path' module to handle file paths
const fs = require('fs');
// Importing the 'path' module to handle file paths
const path = require('path');
// Defining the port number on which the server will listen
const PORT = 5000;
// Creating an HTTP server that listens for incoming requests
http.createServer((req, res) => {
// Constructing the file path to the 'calculator.html' file
  const filePath = path.join(__dirname, 'calculator.html');
  // Reading the 'calculator.html' file and sending its content as a response
  fs.readFile(filePath, (err, content) => {
    // If there is an error reading the file, send a 500 Internal Server Error response
    if (err) {
        // Log the error to the console
      res.writeHead(500);
      // Send a response indicating that there was a server error
      res.end('Server Error');
    } else {
      // If the file is read successfully, send a 200 OK response with the content of the file
      res.writeHead(200, { 'Content-Type': 'text/html' });
      // Send the content of the 'calculator.html' file as the response
      res.end(content, 'utf-8');
    }
  });
}).listen(PORT, () => {
  // Log a message to the console indicating that the server is running and listening on the specified port   
  console.log(`Server running at http://localhost:${PORT}`);
});