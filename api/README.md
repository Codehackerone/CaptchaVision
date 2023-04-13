# API Reference
This API provides a captcha recognition service using the LACC model. The following endpoints are available:

### POST /predict
This endpoint takes a captcha image file and returns the predicted captcha text.

Request
```
  Method: POST
  Headers:
  Content-Type: multipart/form-data
  Body:
  captcha: a file containing a captcha image in PNG or JPEG format
  Response
  Status code: 200 OK
  Body:
  captcha_text: a string containing the predicted captcha text
```

### GET /captcha
This endpoint returns a randomly generated captcha image in PNG format.

Request
```
Method: GET
Response
Status code: 200 OK
Headers:
Content-Type: image/png
Body:
PNG image binary data
```

### GET /health
This endpoint provides a health check for the API.

Request
```
  Method: GET
  Response
  Status code: 200 OK
  Body:
  status: a string indicating the status of the API ("ok" if everything is running smoothly)
```

## Run docker instance

This will create a Docker image with the name "capcha-reognizer" based on the instructions in the Dockerfile.

`docker build -t capcha-recognizer .`

This will start a container from the "capcha-recognizer" image and forward traffic from the host machine's port 80 to the container's port 80, where the app is running.
`docker run -p 80:80 capcha-recognizer`


## Deployment
The API is deployed on AWS Instance and can be accessed at []