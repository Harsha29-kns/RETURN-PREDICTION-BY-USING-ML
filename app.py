import pickle
import numpy as  np
import logging
from flask import Flask, request, render_template, session, redirect, url_for, Response, flash, send_file, jsonify
import csv
from io import StringIO
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import plotly.express as px
import plotly
import json
from werkzeug.utils import secure_filename
import os
from flask_migrate import Migrate
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session handling
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Add this line
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure the profile_pictures directory exists
os.makedirs(os.path.join(app.static_folder, 'profile_pictures'), exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    profile_picture = db.Column(db.String(150), nullable=True)  # Add this line

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('register'))
        
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        # Log out the user after registration
        logout_user()
        
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
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
            flash('Login failed. Please check your username and password.', 'danger')
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

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.username = request.form['username']
        current_user.password = request.form['password']
        
        # Handle profile picture upload
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.static_folder, 'profile_pictures', filename))
                current_user.profile_picture = filename
        
        db.session.commit()
        flash('Profile updated successfully', 'success')
        return redirect(url_for('profile'))
    return render_template('profile.html')

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

        # Validate discount_applied
        if not (0 <= input_data["discount_applied"] <= 100):
            return render_template("index.html", prediction_text="Error: Discount Applied must be between 0 and 100")

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
        app.logger.debug(f"Updated history session data: {session['history']}")

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

from io import BytesIO

@app.route("/download_history_excel")
@login_required
def download_history_excel():
    history = session.get("history", [])
    df = pd.DataFrame(history)
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='History')
    writer.close()  # Correct method to save the Excel file
    output.seek(0)
    return send_file(output, download_name="history.xlsx", as_attachment=True)

@app.route('/download_history_pdf')
@login_required
def download_history_pdf():
    history = session.get("history", [])
    if not history:
        flash("No history available to download.", "warning")
        return redirect(url_for("history"))

    pdf_path = os.path.join(app.static_folder, 'history.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 100, "Prediction History")
    c.drawString(100, height - 120, "Product Price, Discount Applied, Shipping Time, Order Quantity, Prediction")

    y = height - 140
    for entry in history:
        line = f"{entry['Product Price']}, {entry['Discount Applied']}, {entry['Shipping Time']}, {entry['Order Quantity']}, {entry['Prediction']}"
        c.drawString(100, y, line)
        y -= 20

    c.save()
    return send_file(pdf_path, as_attachment=True)

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
    try:
        history = session.get('history', [])
        app.logger.debug(f"History session data: {history}")
        if not history:
            return render_template('visualize.html', graphJSON=None, message="No data available for visualization.")
        
        df = pd.DataFrame(history)
        app.logger.debug(f"DataFrame columns: {df.columns}")
        app.logger.debug(f"DataFrame head: {df.head()}")
        if df.empty or 'Product Price' not in df.columns or 'Order Quantity' not in df.columns or 'Prediction' not in df.columns:
            return render_template('visualize.html', graphJSON=None, message="Insufficient data for visualization.")
        
        fig = px.bar(df, x='Product Price', y='Order Quantity', color='Prediction', title='Prediction History')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        app.logger.debug(f"Generated graph JSON: {graphJSON}")
        return render_template('visualize.html', graphJSON=graphJSON)
    except Exception as e:
        app.logger.error(f"Error in visualize route: {e}")
        return "An error occurred", 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        input_data = {
            "product_price": float(data["product_price"]),
            "discount_applied": float(data["discount_applied"]),
            "shipping_time": int(data["shipping_time"]),
            "order_quantity": int(data["order_quantity"])
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

        return jsonify({"prediction": result, "probability": probability})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=port, debug=True)
