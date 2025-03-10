import pickle
import numpy as np
from flask import Flask, request, render_template, session, redirect, url_for, Response, flash
import csv
from io import StringIO
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import plotly.express as px
import plotly
import json

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session handling
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html')

# Load trained model, scaler, and feature list
model = pickle.load(open("return_prediction_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))  # Ensure same features are used

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        # Get form inputs
        input_data = {
            "product_price": float(request.form["product_price"]),
            "discount_applied": float(request.form["discount_applied"]),
            "shipping_time": int(request.form["shipping_time"]),
            "order_quantity": int(request.form["order_quantity"])
        }

        # Compute new features
        input_data["total_order_value"] = input_data["product_price"] * input_data["order_quantity"]
        input_data["discount_percentage"] = (input_data["discount_applied"] / input_data["product_price"]) * 100
        input_data["high_discount"] = 1 if input_data["discount_percentage"] > 30 else 0
        input_data["fast_shipping"] = 1 if input_data["shipping_time"] <= 3 else 0

        # Ensure correct feature order
        features = np.array([[input_data[feature] for feature in feature_columns]])
        features_scaled = scaler.transform(features)

        # Get prediction and probability
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] * 100  # Probability of return

        # Format the result
        result = "Returned" if prediction == 1 else f"Not Returned ({100 - probability:.2f}% probability)"

        # Save to session history
        history = session.get("history", [])
        history.append({
            "Product Price": input_data["product_price"],
            "Discount Applied": input_data["discount_applied"],
            "Shipping Time": input_data["shipping_time"],
            "Order Quantity": input_data["order_quantity"],
            "Prediction": result
        })
        session["history"] = history

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

@app.route("/history")
@login_required
def history():
    history = session.get("history", [])
    return render_template("history.html", history=history)

@app.route("/clear_history")
@login_required
def clear_history():
    session.pop("history", None)
    return redirect(url_for("history"))

@app.route("/download_history")
@login_required
def download_history():
    history = session.get("history", [])
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["Product Price", "Discount Applied", "Shipping Time", "Order Quantity", "Prediction"])
    cw.writerows([[entry["Product Price"], entry["Discount Applied"], entry["Shipping Time"], entry["Order Quantity"], entry["Prediction"]] for entry in history])
    output = si.getvalue()
    return Response(output, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=history.csv"})

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            data = pd.read_csv(file)
            print(data.columns)  # Print the column names to verify
            predictions = []
            for _, row in data.iterrows():
                input_data = {
                    "product_price": row["product_price"],
                    "discount_applied": row["discount_applied"],
                    "shipping_time": row["shipping_time"],
                    "order_quantity": row["order_quantity"]
                }
                input_data["total_order_value"] = input_data["product_price"] * input_data["order_quantity"]
                input_data["discount_percentage"] = (input_data["discount_applied"] / input_data["product_price"]) * 100
                input_data["high_discount"] = 1 if input_data["discount_percentage"] > 30 else 0
                input_data["fast_shipping"] = 1 if input_data["shipping_time"] <= 3 else 0
                features = np.array([[input_data[feature] for feature in feature_columns]])
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0][1] * 100
                result = "Returned" if prediction == 1 else f"Not Returned ({100 - probability:.2f}% probability)"
                predictions.append(result)
            return render_template("upload_result.html", predictions=predictions)
    return render_template("upload.html")

@app.route('/visualize')
@login_required
def visualize():
    history = session.get('history', [])
    df = pd.DataFrame(history)
    fig = px.bar(df, x='Product Price', y='Order Quantity', color='Prediction', title='Prediction History')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('visualize.html', graphJSON=graphJSON)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
