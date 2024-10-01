import csv
import datetime
from datetime import datetime as dt
from flask import Flask, jsonify, request, redirect, render_template, url_for, flash 
import os
import pickle
import pandas as pd
import numpy as np
import torch
import xgboost as xgb  # XGBoost classifier
from sklearn import preprocessing 
import nltk  # Natural Language Toolkit (NLTK) for text processing
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting the data and performing grid search for hyperparameter tuning
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text data into TF-IDF features
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
from sklearn.pipeline import Pipeline  # For creating machine learning pipelines
from sklearn.metrics import classification_report, accuracy_score  # For evaluating model performance
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
le = preprocessing.LabelEncoder()
from sklearn.preprocessing import LabelEncoder 
from transformers import BertTokenizer, BertModel  # For BERT embeddings



app = Flask(__name__)


# load models
single_response_model = pickle.load(open('single_response_model.pkl', 'rb'))
#matrix_response_model = 'matrix_response_model.pkl'
#matrix_label_encoders = 'matrix_label_encoders.pkl'

with open('matrix_response_model.pkl', 'rb') as matrix_model_file:
    matrix_response_model = pickle.load(matrix_model_file)
with open('matrix_label_encoders.pkl', 'rb') as matrix_encoders_file:
    matrix_label_encoders = pickle.load(matrix_encoders_file)

# Initialize NLTK resources
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


class SavedFile:
    def __init__(self, file_id: int, name: str, path: str, datetime_uploaded: datetime):
        self.id = file_id
        self.name = name
        self.path = path
        self.datetime_uploaded = datetime_uploaded

    def __repr__(self):
        return f"SavedFile(id={self.id}, name='{self.name}', path='{self.path}', datetime_uploaded={self.datetime_uploaded})"



@app.route("/")
def home():
        return render_template(
        "upload.html"
    )

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

# this route is used to upload a file. The file is uploaded into the project 'uploads' folder. The file will be stored on the server
# and the necessary data related to the file such as filename, id etc. in the database. 
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect('/')
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        flash(f'File {file.filename} uploaded successfully!')
        #create saved file object
        saved_file = SavedFile(file_id= "HAI_" + dt.now(), name="placeholder", path=file_path,datetime_uploaded=dt.now())
        selected_option = request.form.get('options')
        #TODO determine which model has been selected ^ 

        #TODO - add this to the database
        return redirect('/')

# this route is used to take manually entered date and save it into a txt file. The file is uploaded into the project 'uploads' folder. The file will be stored on the server
# and the necessary data related to the file such as filename, id etc. in the database. 
@app.route('/manual', methods=['POST'])
def manual_entry():
    # get the data inputted in the form 
    manual_data = request.form.get('manualData')
    # check data is present
    if not manual_data: 
        flash('No data entered')
        return redirect('/')
    
    # Create txt file with the submitted manual data - filename is generated automatically 
    txt_filename = f'manual_data_{dt.now().strftime("%Y%m%d_%H%M%S")}.txt'
    txt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

    # Write the manual data to the txt file
    with open(txt_filepath, 'w') as txt_file:
        txt_file.write(manual_data)

    flash(f'File saved as {txt_filename}')

    if request.form.get('manualOptions') == "single":
        input = [sentence.strip() for sentence in manual_data.split('\n') if sentence.strip()] # split the input into an array
    else:
        input = manual_data

    result = run_model(input, request.form.get('manualOptions'))

    # passing result from model to result page
    return render_template('result.html', output=result)


if __name__ == '__main__':
    app.run(debug=True)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    return ' '.join(tokens)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def embed_text(text):
    """
    Converts a given text into BERT embeddings.

    Args:
    text (str): The input text to embed.

    Returns:
    np.ndarray: A numpy array representing the BERT embedding of the input text.
    """
    print("starting embed_text method")
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = bert_model(**inputs)
    return outputs.pooler_output.detach().numpy().flatten()  # Return the flattened pooler output



def run_model(input, selected_model):
   if selected_model == "single":
    return run_single_model(input) 
   
   if selected_model == "target":
       return run_target_model(input)
   
   if selected_model == "matrix":
        return run_matrix_model(input)

    


def run_matrix_model(input):
    print("matrix method hit")

    # Preprocess the job description (convert to BERT embeddings)
    X_bert = np.array([embed_text(input)])
    # Initialize dictionary to store predictions
    predictions = {}
    # Loop over each question to predict the response
    for question in matrix_label_encoders.keys():
        # Make predictions using the loaded model
        prediction = matrix_response_model.predict(X_bert)
        # Decode the prediction using the corresponding label encoder
        decoded_prediction = matrix_label_encoders[question].inverse_transform(prediction)[0]
        
        # Store the prediction
        predictions[question] = decoded_prediction
    

    # Return the predictions as a JSON response
    print(predictions)
    return predictions
    



def run_linear_model(input):
    return "null"

def run_target_model(input):    
    with open('target_response_model.pkl', 'rb') as file:
        model_components = pickle.load(file)

    model = model_components['model']
    vectorizer = model_components['vectorizer']
    label_encoders = model_components['label_encoders']

    job_description_vectorized = vectorizer.transform([input])
    predictions = model.predict(job_description_vectorized)

    response_columns = ['Question 7', 'Question 8', 'Question 9', 'Question 10', 'Question 11']
    predictions_df = pd.DataFrame(predictions, columns=response_columns)

    for col in response_columns:
        predictions_df[col] = label_encoders[col].inverse_transform(predictions_df[col])

    predictions_adjusted = adjust_predictions(predictions_df.copy())

    decoded_predictions = {}
    for col in response_columns:
        decoded_predictions[col] = str(predictions_adjusted[col].values[0])

    decoded_predictions = convert_target_results(decoded_predictions)
    return decoded_predictions

def adjust_predictions(predictions):
    for index, row in predictions.iterrows():
        main_focus_found = False
        for col in reversed(predictions.columns):
            if main_focus_found:
                predictions.at[index, col] = 'D'  
            if row[col] == 'A':
                main_focus_found = True
    return predictions

def convert_target_results(input_dict):
    # Define mapping 
    number_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    converted_dict = {}
    
    for question, value in input_dict.items():
        converted_dict[question] = number_to_letter[int(value)]
    
    return converted_dict


def run_single_model(input):
    label_mapping = {
    'Response A': 0,
    'Response B': 1,
    'Response C': 2,
    'Response D': 3,
    'Response E': 4,  # Adjust these mappings based on your dataset's labels
    'Response F': 5   # Add or remove as necessary
}  

    processed_descriptions = [preprocess_text(desc) for desc in input]  # Preprocess the job descriptions
    predictions = single_response_model.predict(processed_descriptions)  # Predict using the best model
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse the label mapping for readable output
    return [reverse_label_mapping[prediction] for prediction in predictions]  # Return human-readable predictions