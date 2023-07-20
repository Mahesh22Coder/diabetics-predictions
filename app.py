from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
app = Flask(__name__, static_folder='images')

model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        Pregnancies = request.form['Pregnancies'] 
        Glucose = request.form['Glucose'] 
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        Outcome = request.form['Outcome']
       
        arr = np.array([[Pregnancies,Glucose,BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome]])
        return redirect(url_for('after', Pregnancies=Pregnancies, Glucose=Glucose, BloodPressure=BloodPressure, SkinThickness=SkinThickness,Insulin=Insulin,BMI=BMI,DiabetesPedigreeFunction=DiabetesPedigreeFunction,Age=Age,Outcome=Outcome))
    return render_template('index.html')

@app.route("/after")
def after():
    Pregnancies = request.args.get('Pregnancies')
    Glucose = request.args.get('Glucose')
    BloodPressure = request.args.get('BloodPressure')
    SkinThickness = request.args.get('SkinThickness')
    Insulin = request.args.get('Insulin')
    BMI = request.args.get('BMI')
    DiabetesPedigreeFunction = request.args.get('DiabetesPedigreeFunction')
    Age = request.args.get('Age')
    Outcome= request.args.get('Outcome')

    arr = np.array([[Pregnancies,Glucose,BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome]])
    pred = model.predict(arr)[0]

    return render_template('after.html', Pregnancies=Pregnancies, Glucose=Glucose, BloodPressure=BloodPressure, SkinThickness=SkinThickness,Insulin=Insulin,BMI=BMI,DiabetesPedigreeFunction=DiabetesPedigreeFunction,Age=Age,Outcome=Outcome )

if __name__ == "__main__":
    app.run(debug=True)