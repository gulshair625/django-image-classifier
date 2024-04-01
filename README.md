# Image Classification

This repository provides a Flask application that classifies images into different categories using a pre-trained model from the Hugging Face Model Hub. The application is containerized with Docker, making it easy to deploy and scale.

## Table of Contents

- [Tech Stack](#tech-stack)
- [Project Setup](#project-setup)
- [Running the Application](#running-the-application)
- [Interacting with the Application](#interacting-with-the-application)
- [Additional Resources](#additional-resources)
- [About the Author](#about-the-author)

## Tech Stack

- Flask 3.0.2
- Python 3.12.2
- PyTorch 2.2.0
- Transformers 4.38.2
- Docker 25.0.4

## Project Setup

### 1. Build the Docker Image

Navigate to the project directory and build the Docker image using the following command:

```bash
docker build -t flask_docker .
```

### 2. Run the Docker Container

After building the Docker image, run the container with the following command:

```bash
docker run -d -p 6000:6000 flask_docker
```

This command runs the `flask_docker` image as a container in detached mode (`-d`) and maps port 6000 of the container to port 6000 on the host machine.

## Running the Application

The Flask application is designed to classify images based on their content. It uses a pre-trained model from the Hugging Face Model Hub for image classification. The model is loaded with the following code snippet:

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor

repo_name = "yangy50/garbage-classification"
model = AutoModelForImageClassification.from_pretrained(
    repo_name,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
inference_model = model
image_processor = AutoImageProcessor.from_pretrained(repo_name)
```

The application exposes a `/predict` endpoint that accepts POST requests with an image file in form data. The key for the image file is `image`.

## Interacting with the Application

To interact with the Flask application, send a POST request to the exposed port (6000) with an image file in form data. Use the key `image` for the image file.

You can use tools like `curl` or Postman to send the request. Here's an example using `curl`:

```bash
curl -X POST -H "Content-Type: multipart/form-data" -F "image=@path/to/your/image.jpg" http://172.17.0.2:6000/predict
```

Replace `path/to/your/image.jpg` with the actual path to the image file you want to send.

## API Response Format

The API response is structured in JSON format and includes a status code and a body with detailed information about the classification result. The body contains the following fields:

- `class`: The category to which the image has been classified.
- `probability`: The confidence level of the classification, expressed as a percentage.
- `text`: A message providing additional information about the classification result and the importance of recycling.


## Additional Resources

- [Learn more about Docker and Flask](https://docs.docker.com/samples/flask/)
- [Deploy to production](https://flask.palletsprojects.com/en/3.0.x/deploying/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## About the Author

This project is developed by Gul Shair Shakeel, a Machine Learning Engineer with a strong background in deep learning and image processing. Gul Shair has demonstrated expertise in leveraging transformer models for image classification tasks, as evidenced by his work on the garbage classification project. He is currently employed at REV9 SOLUTIONS, where he continues to contribute to innovative machine learning solutions.

For more information about Gul Shair's work and achievements, please visit his [LinkedIn profile](https://www.linkedin.com/in/gulshair625/).

