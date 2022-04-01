from flask import Flask,render_template,request
import joblib
import numpy as np

model=joblib.load('heart_risk_prediction_regression_model.sav')

app=Flask(__name__) #application

@app.route('/')
def index():

	return render_template('index.html')

@app.route('/getresults',methods=['POST'])
def getresults():

	result=request.form 

	name=result['name']
	gender=float(result['gender'])
	age=float(result['age'])
	tc=float(result['tc'])
	hdl=float(result['hdl'])
	sbp=float(result['sbp'])
	smoke=float(result['smoke'])
	bpm=float(result['bpm'])
	diab=float(result['diab'])

	test_data=np.array([gender,age,tc,hdl,smoke,bpm,diab]).reshape(1,-1)

	prediction=abs(model.predict(test_data))

	# if prediction < 0:
	# 	prediction = 0
	#prediction = float(prediction)
	#prediction = max(prediction,key=0)
	#print(type(prediction))

	resultDict={"name":name,"risk":round(prediction[0][0],2),"tc":tc,"hdl":hdl,"sbp":sbp}

	return render_template('result.html',results=resultDict)

app.run(debug=True)