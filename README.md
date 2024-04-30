Emotion Recognition System

This project provides a web application that recognizes human emotions from facial expressions using uploaded images or live webcam feed. The application is built using Python, Flask, and machine learning models trained on diverse facial datasets.

Getting Started

These instructions will guide you on how to set up and run the Emotion Recognition System on your local machine for development and testing purposes.

Prerequisites

Before you begin, ensure you have the following installed:

Python 3.8 or higher

pip (Python package installer)

Installation

Clone the Repository

First, clone this repository to your local machine using Git.

git clone https://github.com/your-username/emotion-recognition.git

cd emotion-recognition

Install Required Packages.

Install all the required packages using pip. It's recommended to use a virtual environment.

python -m venv venv

source venv/bin/activate 

On Windows use `venv\Scripts\activate`

pip install -r requirements.txt

Running the Application
Start the ApplicationRun the app.py file to start the Flask server.
python app.py

Access the ApplicationOpen your web browser and visit http://localhost:5000. 
The web interface should be accessible now.

Using the Application

Upload Image: Click on the "Upload Image" button to select an image file from your computer. The system will analyze the image and display the detected emotion.

Webcam Feed: Click on the "Start Webcam" button to open the webcam feed. The system will analyze the feed in real-time and display the detected emotions.

Features

Single image emotion recognition
Real-time emotion recognition using webcam

Contributing

We welcome contributions to improve this project. Please feel free to fork the repository, make improvements, and submit pull requests.

