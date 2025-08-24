from flask import Flask,render_template,request

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route("/")
def homepage():
    return render_template("home.html")

@app.route("/predictdata",methods=['GET','POST'])
def predictionpage():
    if request.method=="GET":
        return render_template("prediction.html")
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid prediction")
        results=predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template("prediction.html",results=results[0])
        
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
