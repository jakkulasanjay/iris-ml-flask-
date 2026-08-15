from  flask import request,render_template,Flask 
from flask import *

import joblib
model = joblib.load("iris_prediction.pkl")
app=Flask(__name__)
@app.route('/',methods=["GET","POST"])

#model =joblib.load("irus_prediction.pkl")

def home():

    if request.method == "POST":
        sepal_length=float(request.form['sepal_length'])
        sepal_width=float(request.form['sepal_width'])

        petal_length=float(request.form['petal_length'])       
        petal_width=float(request.form['petal_width'])

        names=["setosa", "versicolor", "virginica"]

        prediction=model.predict([[
            sepal_length,sepal_width,petal_length,petal_width
        ]])
        result=names[int(prediction[0])]
        #prediction[0] because prediction returns array so we start with 1 st value
        return render_template("iris_prj.html",prediction=result, image=result + ".jpg")

    return render_template('iris_prj.html')



if __name__ =="__main__":
    app.run(debug=True)

