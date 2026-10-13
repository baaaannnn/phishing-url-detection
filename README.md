# Phishing URL Detection

Academic project for the Big Data Analytics course. Classifies URLs as legitimate or phishing by extracting structural features and comparing two machine learning models.

## Features
- Extracts URL length and counts of dots, hyphens, and @ characters.
- Normalizes features using MinMaxScaler.
- Visualizes the features with parallel coordinates.
- Trains a Decision Tree and K-Nearest Neighbours classifier with k=7.
- Uses a 66% training and 34% testing split with random_state=42.
- Prints accuracy, precision, recall, and F1-score for both models.
- Visualizes the trained decision tree and compares classifier accuracy.

## Technologies
Python, pandas, scikit-learn, Matplotlib.

## Run
```bash
python -m venv .venv
python -m pip install -r requirements.txt
python project.py
```
Activate the virtual environment before installing packages. Run from the repository directory so URLDataset.csv is available. Close each chart window to continue to the next step.

## Files
- project.py: supplied academic implementation.
- URLDataset.csv: small URL dataset supplied with the project.
- requirements.txt: Python dependencies.

## Method
The course implementation normalizes the complete dataset before splitting. The model comparison is an academic demonstration on this dataset; it is not a deployed website-security service. Original dataset publication and licensing details were not supplied with the course files.

This is an academic team project. This portfolio copy does not attribute all implementation work to one contributor.
