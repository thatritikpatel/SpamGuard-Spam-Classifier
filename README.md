# SpamGuard-Spam-Classifier

SpamGuard-Spam-Classifier is a machine learning-based solution designed to classify SMS messages as either spam or legitimate (ham). This project leverages natural language processing (NLP) techniques and machine learning models to build an efficient spam classifier. The classifier is deployed via a Flask web application for easy interaction and real-time predictions.

## Table of Contents

- [Introduction](#introduction)
- [What is a Spam Classifier?](#what-is-a-spam-classifier)
- [Problem Solved by the Spam Classifier](#problem-solved-by-the-spam-classifier)
- [Challenges of Not Using a Spam Classifier](#challenges-of-not-using-a-spam-classifier)
- [Key Benefits](#key-benefits)
- [Features](#features)
- [Dataset](#dataset)
  - [About the Dataset](#about-the-dataset)
  - [Data Sources](#data-sources)
  - [Using the Dataset](#using-the-dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Flask Application](#running-the-flask-application)
  - [Making Predictions](#making-predictions)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Introduction

SpamGuard-Spam-Classifier uses machine learning and natural language processing (NLP) techniques to detect spam messages accurately. It provides an easy-to-use interface to classify messages in real-time. Built using Python and a variety of data science libraries, the classifier predicts whether a message is spam or not based on its content. This project also includes a web application for convenient access and analysis.

## What is a Spam Classifier?

A spam classifier is a machine learning model trained to distinguish between "spam" and "ham" messages (legitimate messages). It analyzes the content of incoming messages and classifies them based on patterns identified from a labeled dataset. 

In this project, the spam classifier is specifically designed to detect unwanted messages in SMS (Short Message Service) text, such as advertising, phishing, and other unsolicited content.

## Problem Solved by the Spam Classifier

SpamGuard-Spam-Classifier addresses the following issues:

1. **Inbox Clutter**: Spam messages take up valuable space in users' inboxes, making it harder to focus on important communications.
2. **Security Risks**: Spam messages often contain phishing attempts, malware links, or scams, posing a security threat to users.
3. **Loss of Productivity**: Users waste time manually sorting through spam, reducing overall productivity.
4. **System Strain**: Unfiltered spam increases storage and bandwidth requirements, putting a strain on system resources.

By detecting and filtering out spam messages automatically, this classifier helps save time, improve productivity, and enhance the overall user experience.

## Challenges of Not Using a Spam Classifier

Without a spam classifier, users and systems face several challenges:

- **Time-Consuming Manual Sorting**: Users must manually filter through spam messages, which wastes valuable time and effort.
- **Increased Risk of Malicious Content**: Spam messages can contain phishing attacks, malware, or scams, which can compromise personal and financial data.
- **Higher Resource Consumption**: Without filtering, systems must process and store unnecessary spam messages, increasing costs and reducing efficiency.
- **Negative User Experience**: Continuous spam exposure can lead to frustration and a negative experience with messaging platforms or services.

## Key Benefits

- **Accurate Spam Detection**: The classifier uses machine learning models, ensuring high accuracy in identifying spam messages.
- **Real-Time Predictions**: The classifier can predict whether a message is spam or not in real time via a web interface.
- **User-Friendly Interface**: The application provides a simple and intuitive interface for users to interact with.
- **Visual Insights**: The project includes visualizations to help users understand the dataset and model performance better.
- **Improved Productivity**: By automating spam detection, users spend less time managing unwanted messages, leading to better focus and increased productivity.

## Features

- **Spam Classification**: Classifies SMS messages as spam or ham (legitimate).
- **Flask Web Interface**: A web application that provides a simple interface for message classification.
- **Model Performance Visualizations**: Graphs and metrics that show the performance of the trained model.
- **Real-Time Prediction**: A REST API allows users to classify messages instantly through the web interface.
- **Easy Integration**: The model can be deployed on any platform supporting Python and Flask.

## Dataset

The SMS Spam Collection dataset, used to train and evaluate the classifier, contains 5,574 labeled messages (either spam or ham). The dataset consists of SMS messages in English, each labeled either as "spam" (unsolicited messages) or "ham" (legitimate messages).

### About the Dataset

- **Size**: 5,574 SMS messages (spam and ham).
- **Columns**:
  - **v1 (Label)**: Labels the message as either `ham` or `spam`.
  - **v2 (Message)**: The actual text content of the SMS message.

### Data Sources

The dataset was created from various sources:

- **Grumbletext Website**: 425 spam messages collected from a UK-based forum where users report spam SMS.
- **NUS SMS Corpus**: 3,375 legitimate SMS messages from Singaporean university students.
- **Caroline Tag’s PhD Thesis**: 450 legitimate messages used for research.
- **SMS Spam Corpus v.0.1**: 1,002 legitimate messages and 322 spam messages collected for research.

### Using the Dataset

1. **Preprocessing**: Clean the text data by removing special characters, numbers, and irrelevant words. Optionally, apply stemming or lemmatization.
2. **Feature Extraction**: Use TF-IDF or Bag of Words techniques to convert text messages into numerical features.
3. **Model Selection and Training**: Choose algorithms like Naive Bayes, SVM, or Logistic Regression and train the model on the preprocessed data.
4. **Evaluation**: Evaluate the trained model using accuracy, precision, recall, and F1-score metrics.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Steps to Set Up the Project

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/SpamGuard-Spam-Classifier.git
   cd SpamGuard-Spam-Classifier
   ```

2. **Create a Virtual Environment**:

   It's recommended to create a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   Install the necessary Python libraries.

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Flask Application

To start the web application, run the following command:

```bash
python app.py
```

Once the server is running, you can access the application in your web browser at:

```
http://127.0.0.1:5000
```

### Making Predictions

- Use the web interface to enter a message and get an immediate prediction (spam or ham).
- The system will process the input and return the result.

## Technologies

This project uses the following technologies:

- **Python**: The core programming language for the project.
- **Flask**: A micro web framework to create the web application.
- **scikit-learn**: A machine learning library for model training and evaluation.
- **pandas**: Data manipulation and analysis.
- **nltk**: Natural Language Toolkit for text preprocessing.
- **TfidfVectorizer**: For converting text into numerical features.
- **matplotlib** & **seaborn**: For data visualization.
- **wordcloud**: To visualize the most frequent terms in the dataset.

## Project Structure

```
SpamGuard-Spam-Classifier/
├── app.py               # Main application file for the Flask app
├── static/              # Static files (CSS, JavaScript, Images)
├── templates/           # HTML templates for the web interface
├── models/              # Folder for saved machine learning models
├── data/                # Folder containing the dataset
├── notebooks/           # Jupyter notebooks for experiments
├── requirements.txt     # List of required Python libraries
└── README.md            # Project documentation (this file)
```

- `app.py`: Main Flask app for the user interface and prediction functionality.
- `models/`: Contains saved models for quick prediction.
- `notebooks/`: Contains Jupyter notebooks for data analysis and experimentation.

## Model Training

The machine learning models used for the spam classifier are trained using scikit-learn. The models are saved in the `models/` directory and can be used to make real-time predictions via the Flask app.

## Contributing

Contributions are welcome! To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -am 'Add Your Feature'`).
4. Push your changes to your fork (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details."# SpamGuard-Spam-Classifier" 
