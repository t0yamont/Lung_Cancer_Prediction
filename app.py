import torch
import torch.nn as nn
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import logging

# Initialize the Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = '050403'  # Replace with a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Optional but recommended

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'  # Ensure this is set
login_manager.init_app(app)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    input_data = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))


    # Define the neural network structure (must match the trained model)
class LungCancerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LungCancerModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# Load the model
input_size = 23  # Number of features
num_classes = 3  # 'Low', 'Medium', 'High'
model = LungCancerModel(input_size, num_classes)
model.load_state_dict(torch.load(r'C:\Users\monti\OneDrive\Área de Trabalho\UNI\lasttry\best_lung_cancer_model.pth'))
model.eval()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

# Sign up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user exists
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists')
            return redirect(url_for('signup'))

        # Create new user
        new_user = User(username=username, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully. Please log in.')
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Authenticate user
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))

    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    features_list = [
        'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards',
        'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity',
        'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood',
        'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing',
        'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold',
        'Dry Cough', 'Snoring'
    ]

    prediction = None

    if request.method == 'POST':
        try:
            Age = float(request.form['Age'])
            Gender = request.form['Gender']
            # Map Gender to numeric value
            gender_mapping = {'Female': 0, 'Male': 1}
            Gender = gender_mapping.get(Gender, 0)

            inputs = [Age, Gender]

            for feature in features_list:
                value = float(request.form[feature])
                if 1 <= value <= 8:
                    inputs.append(value)
                else:
                    flash(f"Invalid input for {feature}. Please enter a value between 1 and 8.")
                    return render_template('predict.html', features=features_list)

            # Convert inputs to the format expected by the model
            inputs_tensor = torch.tensor([inputs], dtype=torch.float32)

            # Make the prediction
            with torch.no_grad():
                output = model(inputs_tensor)
                _, predicted = torch.max(output.data, 1)
                level_mapping = {0: 'High', 1: 'Low', 2: 'Medium'}
                predicted_level = level_mapping.get(predicted.item(), 'Unknown')

            # Save prediction to database
            prediction_record = Prediction(user_id=current_user.id, input_data=str(inputs), result=predicted_level)
            db.session.add(prediction_record)
            db.session.commit()

            prediction = predicted_level


        except Exception as e:
            logging.error("Error in input: %s", e)
            flash("Error in input. Please ensure all fields are filled correctly.")
            return render_template('predict.html', features=features_list)

    return render_template('predict.html', features=features_list, prediction=prediction)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)