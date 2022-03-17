import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data = {'cough': 0, 'fever': 0, 'sore_throat': 0, 'shortness_of_breath': 0,
     'head_ache': 0, 'age_60_and_above': 0, 'test_indication': 0, 
     'gender_Other': 0,'gender_female': 0, 'gender_male': 0}
    df = pd.DataFrame(data, index = [1])

    gender = request.values['gender']
    df.loc[1, 'gender_'+gender] = 1

    senior = request.values['senior']
    df.loc[1, 'age_60_and_above'] = int(senior)

    if request.values['cough']:
        df.loc[1, 'cough'] = 1
    if request.values['fever']:
        df.loc[1, 'fever'] = 1
    if request.values['sore']:
        df.loc[1, 'sore_throat'] = 1
    if request.values['breath']:
        df.loc[1, 'shortness_of_breath'] = 1
    if request.values['head']:
        df.loc[1, 'head_ache'] = 1

    indication = request.values['Indication']
    df.loc[1, 'test_indication'] = int(indication)

    prediction = model.predict(df).item()
    if prediction == 1:
        result = '''
        There is a chance that you are Covid Positive!
        Please consider Taking an RTPCR test to confirm!
        Meanwhile maintain social distancing.
        Stay Safe!
        '''
    else:
        result = '''
        You are Covid Negative based on the information provided.
        Still, please maintain social distancing and other saftey measures.
        Stay Safe!
        '''
        
    
    return render_template('result.html', prediction_text = f'{result}')

if __name__ == '__main__':
    app.run()
