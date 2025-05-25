# 📈 Return Prediction Using Machine Learning

This project uses machine learning to predict the return status of products based on synthetic e-commerce transaction data. It includes model training, data preprocessing, and a web-based interface for making predictions using a trained ML model.

## 🔗 Live Demo

Access the deployed application here: [pre-1-feve.onrender.com](https://pre-1-feve.onrender.com)

## 📁 Project Structure
RETURN-PREDICTION-BY-USING-ML/ │ 
├── app.py                         # Flask web application
├── train_model.py                 # Script to train the machine learning model
├── synthetic_returns_data_updated.csv  # Synthetic dataset 
├── return_prediction_model.pkl    # Trained model saved as pickle file
├── feature_columns.pkl            # Feature columns saved for model input
├── scaler.pkl                     # StandardScaler object for feature scaling
├── templates/ │   └── index.html                 # HTML page for user input and prediction result
├── static/ │   └── style.css                  # CSS styles for the frontend 
├── requirements.txt               # List of Python dependencies 
├── Procfile                       # For deploying on platforms like Render
├── runtime.txt                    # Specifies Python version for deployment
└── README.md                      # Project documentation

## ⚙️ Installation

1. **Clone the Repository:**

bash
git clone https://github.com/Harsha29-kns/RETURN-PREDICTION-BY-USING-ML.git
cd RETURN-PREDICTION-BY-USING-ML

2. Create and Activate Virtual Environment (optional but recommended):



python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install the Dependencies:

pip install -r requirements.txt

🧠 Model Training

To train the machine learning model using the synthetic dataset:

python train_model.py

This will create three files:

return_prediction_model.pkl (trained model)

feature_columns.pkl (feature column names)

scaler.pkl (feature scaler)


🚀 Running the Web App Locally

After training the model, run the Flask app:

python app.py

Open your browser and go to: http://localhost:5000

Enter product data and receive a prediction on whether the item is likely to be returned.

📦 Deployment

This project is configured for deployment on platforms like Heroku or Render using:

Procfile

runtime.txt


🧰 Features

Preprocessing of real-world-like synthetic data

Logistic regression-based return prediction

Web interface using Flask and HTML/CSS

Model serialization using pickle

Real-time prediction on deployed site


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing

Contributions, issues and feature requests are welcome!

Fork this repo

Create a new branch (git checkout -b feature-branch)

Make your changes

Commit and push (git commit -am 'Add feature' && git push origin feature-branch)

Submit a pull request


📬 Contact

For questions or suggestions, feel free to reach out via GitHub or email: chilukuriharsha116@gmail.com
