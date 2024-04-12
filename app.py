import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# Create flask app
app = Flask(__name__)

# Load predictive model
model = pickle.load(open('xgboost_model.pkl', 'rb'))

# Define functions for feature engineering
def calculate_EGP(float_features):
    # Assuming 3P Made is at index 6 and 3P Attempted is at index 7
    three_pointers_made = float_features[6]
    field_goals_made = float_features[3]
    field_goals_attempted = float_features[4]
    three_pointers_attempted = float_features[7]
    
    # Calculate Effective Goals Percentage
    try:
        EGP = (field_goals_made + 1.5 * three_pointers_made) / (field_goals_attempted + three_pointers_attempted)
    except:
        EGP = 0
    return EGP

def calculate_EFF(float_features):
    # Assuming points made, assists, and turnovers are at indices 2, 15, and 18 respectively
    points_made = float_features[3]
    assists = float_features[15]
    turnovers = float_features[18]
    minutes_played = float_features[1]
    
    # Calculate Efficiency
    try:
        EFF = (points_made + assists - turnovers) / minutes_played
    except:
        EFF = 0
    return EFF


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    # Perform feature engineering
    EGP = calculate_EGP(float_features)
    EFF = calculate_EFF(float_features)
    
    # Append the engineered features to the final features array
    final_features[0] = np.append(final_features[0], [EGP, EFF])
    print(final_features)
    prediction = int(model.predict(final_features))
    if prediction==1:
        output = True
    else:
        output = False
    print(prediction)
    
    return render_template('index.html', prediction_text='Will this player play for more than 5 years at the NBA? {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)