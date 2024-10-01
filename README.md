# HERA AI Application

This appliation has been developed for the deployment of a Machine Learning model which will assist in the grading of higher education roles under the HERA framework

## Installation
The application will need to be run locally in a flask virtual environment

#### Prerequisites
- VSCode
- Python installed on your machine

#### Reproduce environment and run project
- Open the project in VSCode
- In VSCode, open the command palette using Ctrl+Shift+P and select *Python: Create Environment*. Select *venv* and the Python environment you wish to use.
- Open a VSCode terminal and activate the virutal environment:

```bash
.venv\Scripts\activate
```

 within the .venv folder, install flask:

```bash
pip install flask
```

- You will also need to refer to the imports at the top of app.py and install all of the required libraries using pip install (pandas, torch, xgboost, nltk, transformers). Refer to any imports that are underlined as these will need installing in your environment.

- After doing this, you should now be able to run the application by navigating back to the root folder and using:

```bash
python -m flask run
```
- You will now be prompted to navigate to the localhost in the terminal

## Using the application
The application is made up of a simple interface which allows the user to input a job description via file upload or manual text entry, and run it against either of the four models for each type of grading.