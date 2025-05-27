from flask import Flask, render_template, request, render_template_string
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Создание экземпляра Flask
app = Flask(__name__)

# Загрузка моделей
with open('y_pred_1.pkl', 'rb') as f1:
    loaded_model_1 = pickle.load(f1)

with open('y_pred_2.pkl', 'rb') as f2:
    loaded_model_2 = pickle.load(f2)

# Определение главной страницы
@app.route('/', methods=["GET", "POST"])
def index():
    message = ""
    if request.method == "POST":
        try:
            # Здесь мы получаем данные из формы
            density = float(request.form["density"])          
            modulus = float(request.form["modulus"])          
            hardener_amount = float(request.form["hardener_amount"])  
            epoxy_content = float(request.form["epoxy_content"])      
            flash_point = float(request.form["flash_point"])           
            surface_density = float(request.form["surface_density"])   
            resin_consumption = float(request.form["resin_consumption"])  
            weaving_angle = float(request.form["weaving_angle"])       
            weaving_step = float(request.form["weaving_step"])         
            weaving_density = float(request.form["weaving_density"])   

            # Собираем массив с параметрами для модели
            input_data = [
                density,
                modulus,
                hardener_amount,
                epoxy_content,
                flash_point,
                surface_density,
                resin_consumption,
                weaving_angle,
                weaving_step,
                weaving_density
            ]

            # Прогоняем через функции предсказания
            ypred1 = predict1(input_data)
            ypred2 = predict2(input_data)
           
            # Формируем сообщение для отображения
            message = (
                f"Параметр 'Модуль упругости при растяжении, ГПа': {ypred1}<br>"
                f"Параметр 'Прочность при растяжении, МПа': {ypred2}"
            )
        except Exception as e:
            message = f"Ошибка вычисления: {str(e)}"

    return render_template("index.html", message=message)

# Вспомогательные функции предсказания
def predict1(input_data):
    y_pred1 = loaded_model_1.predict([input_data])[0]
    return y_pred1

def predict2(input_data):
    y_pred2 = loaded_model_2.predict([input_data])[0]
    return y_pred2

# Запуск приложения
if __name__ == "__main__":
    app.run(debug=True)
