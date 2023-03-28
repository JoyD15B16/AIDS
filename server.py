from flask import Flask, render_template,request
import car_prediction
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    production_year = request.args.get('production_year') 
    category = request.args.get('category')
    leather_interior = request.args.get('leather_interior')
    fuel_type = request.args.get('fuel_type')
    mileage = request.args.get('mileage')
    cylinders = request.args.get('cylinders')
    gear_box_type = request.args.get('gear_box_type')
    wheel =  request.args.get('wheel')
    airbags = request.args.get('airbags')
    price = car_prediction.predict_price(
        Production_year = production_year,
        Category = category,
        Leather_interior = leather_interior,
        Fuel_type = fuel_type,
        Mileage = mileage,
        Cylinders = cylinders,
        Gear_box_type = gear_box_type,
        Wheel = wheel,
        Airbags = airbags,
        )
    return render_template('result.html',res=price)


if __name__ == '__main__':
    app.run(debug=True)