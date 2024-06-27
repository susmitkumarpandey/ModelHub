from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, IntegerField, DateField
from wtforms.validators import DataRequired, ValidationError
from datetime import date
import pandas as pd
import numpy as np
import pickle

cars = pd.read_csv('Cleaned_Car_data.csv')
models = cars['name'].unique()
companies = cars['company'].unique()
# audi_cars = cars[cars['company'] == 'Audi']
# print(audi_cars)

app = Flask(__name__)
app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'
bootstrap = Bootstrap5(app=app)

# If you want to use FlaskForm ,the use this code:

# def validate_date(form, field):
#     if field.data is not None and field.data > date.today().year:
#         raise ValidationError(
#             'Year must be less than or equal to current year.')


# class predictform(FlaskForm):
#     company = SelectField(
#         'Car Company', validate_choice=True, choices=companies)
#     model = SelectField('Car Model', validate_choice=True, choices=models)
#     year = IntegerField('Year of Purchase', validators=[
#                         DataRequired(), validate_date])
#     km = IntegerField('Distance Travelled', validators=[DataRequired()])
#     ftype = SelectField('Fuel Type', validate_choice=True,
#                         choices=['Petrol', 'Diesel'])
#     submit = SubmitField('Submit')


predic = pickle.load(open('LinearRegressionModel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=["GET", "POST"])
def predict_model():
    # form = predictform()
    # if form.validate_on_submit():
    # prediction = pipe.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array(
    #     [form.model.data, form.company.data, form.year.data, form.km.data, form.ftype.data]).reshape(1, 5)))
    # return render_template('predict.html', prediction_text='Predicted Price: Rs. {}'.format(prediction), form=form)
    if request.method == "POST":
        model = request.form.get('model')
        company = request.form.get('company')
        year = request.form.get('year')
        distance = request.form.get('dist')
        fuel = request.form.get('ftype')
        prediction = predic.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array(
            [model, company, year, distance, fuel]).reshape(1, 5)))
        return render_template('predict.html', prediction_text='Predicted Price: Rs. {}'.format(prediction), models=models, companies=companies)
    return render_template('predict.html', models=models, companies=companies)


@app.route('/predict2')
def predict2():
    pass


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
