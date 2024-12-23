import os
import pandas as pd
import pickle
import yaml
from flask import Flask, render_template, redirect, url_for, flash, request, session, make_response, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
import numpy as np
from gemini import model, get_suggestion
from flask_sqlalchemy import SQLAlchemy
from models.user import db, User
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from flask_session import Session

# Load environment variables
load_dotenv()

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'flask_session')
app.config['SESSION_PERMANENT'] = False
Session(app)

# Create session directory if it doesn't exist
if not os.path.exists(app.config['SESSION_FILE_DIR']):
    os.makedirs(app.config['SESSION_FILE_DIR'])

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except (ValueError, TypeError):
        return None

# Initialize database tables
def init_db():
    with app.app_context():
        db.create_all()
        # Create admin user if it doesn't exist
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin')
            admin.set_password('password123')
            db.session.add(admin)
            db.session.commit()

# Available models
MODELS = {
    "Linear Regression": "models/dataset1/dataset1_linear_regression_default.pkl",
    "XGBoost": "models/dataset1/dataset1_xgboost_regression_optimized.pkl",
    "Random Forest": "models/dataset1/dataset1_random_forest_regression_optimized.pkl",
    "Ridge Regression": "models/dataset1/dataset1_ridge_regression_optimized.pkl",
}

# Load label encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

def load_model(model_path):
    """
    Load a machine learning model from the given path.
    """
    return pickle.load(open(model_path, "rb"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    """
    Load configuration from YAML file with UTF-8 encoding
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'features.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Load configuration
CONFIG = load_config()
EXPECTED_FEATURES = CONFIG['feature_order']

def process_timetable(file_path):
    """
    Process timetable file.
    Extracts columns: mamh, sotc, hocky, namhoc, khoahoc, hedt, khoa.
    """
    try:
        logger.info(f"Processing timetable file: {file_path}")
        df = pd.read_excel(file_path, header=7)
        # Column indices: B(1), H(7), P(15), Q(16), O(14), R(17), S(18)
        required_indices = [1, 7, 15, 16, 14, 17, 18]
        
        # Check if DataFrame has enough columns
        if df.shape[1] <= max(required_indices):
            logger.error("Uploaded timetable file does not have enough columns.")
            flash('Uploaded timetable file is malformed or does not contain the required columns.', 'danger')
            return []
        
        processed_data = df.iloc[:, required_indices]
        processed_data.columns = ['mamh', 'sotc', 'hocky', 'namhoc', 'khoahoc', 'hedt', 'khoa']
        
        # Ensure any missing column is added with -1
        required_cols = ['mamh', 'sotc', 'hocky', 'namhoc', 'khoahoc', 'hedt', 'khoa']
        for col in required_cols:
            if col not in processed_data.columns:
                processed_data[col] = -1
        
        processed_data = processed_data.fillna(-1)
        logger.info(f"Processed timetable data: {processed_data.head()}")
        return processed_data.to_dict('records')
    
    except Exception as e:
        logger.exception("An error occurred while processing the timetable file.")
        flash('An error occurred while processing the timetable file. Please ensure it is in the correct format.', 'danger')
        return []

def process_student_info(file_path):
    """Process student info file."""
    try:
        logger.info(f"Processing student info file: {file_path}")
        df = pd.read_excel(file_path, header=None)
        
        # Extract gioitinh, lopsh, namsinh
        gioitinh = str(df.iloc[0, 5]).strip()  # Column F1
        lopsh = df.iloc[1, 3]  # Column D2
        namsinh = pd.to_datetime(df.iloc[0, 3], errors='coerce').year
        
        # Initialize noisinh as None to be selected later
        noisinh = 'nan'
        # hedt in d3
        hedt = df.iloc[2, 3]  # Column D3
        #sv_khoa in F2
        khoa = df.iloc[1, 5]  # Column F2
        # Extract dtbhk and sotchk where "Trung bình học kỳ" is present in column C
        dtbhk_sotchk_rows = df[df.iloc[:, 2].str.contains("Trung bình học kỳ", na=False)].index.tolist()
        dtbhk_sotchk = []
        for idx in reversed(dtbhk_sotchk_rows):  # Process rows from bottom to top
            dtbhk = df.iloc[idx, 8]  # Column D
            sotchk = df.iloc[idx, 3]  # Column E
            dtbhk_sotchk.append((dtbhk, sotchk))
        
        # Convert to input format (dtbhk1, dtbhk2, ..., sotchk1, sotchk2, ...)
        dtbhk_sotchk_input = {}
        for i, (dtbhk, sotchk) in enumerate(dtbhk_sotchk, start=1):
            dtbhk_sotchk_input[f'dtbhk{i}'] = dtbhk
            dtbhk_sotchk_input[f'sotchk{i}'] = sotchk
        
        # Define all possible dtbhk and sotchk fields up to dtbhk22 and sotchk22
        for i in range(1, 23):
            dtbhk_sotchk_input.setdefault(f'dtbhk{i}', -1)
            dtbhk_sotchk_input.setdefault(f'sotchk{i}', -1)
        
        # Define other required fields
        other_required_fields = {
            'trangthai': -1,
            'tinhtrang': -1,
            'mamh_tt': -1,
            'namhoc_monhoc': -1,
            'hocky_monhoc': -1,
            'gap_hocky': -1,
            'hocky_monhoc_count': -1,
            'namhoc_monhoc_count': -1
        }
        
        # Fill remaining fields with -1
        remaining_fields = {col: -1 for col in ['mamh', 'malop', 'sotc', 'hocky', 'namhoc', 'khoahoc']}
        
        # Combine all inputs maintaining the order from feature_order
        student_info = {
            'gioitinh': gioitinh,
            'lopsh': lopsh,
            'namsinh': namsinh,
            'noisinh': noisinh,
            'hedt': hedt,
            'khoa': khoa,
            **dtbhk_sotchk_input,
            **other_required_fields,
            **remaining_fields
        }
        
        # Reorder student_info based on CONFIG['feature_order']
        ordered_student_info = {
            field: student_info.get(field, -1) for field in CONFIG['feature_order']
        }
        
        logger.info(f"Processed student info: {ordered_student_info}")
        return ordered_student_info
    except Exception as e:
        logger.exception("An error occurred while processing the student info file.")
        flash('An error occurred while processing the student info file. Please ensure it is in the correct format.', 'danger')
        return {}

def convert_np_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def encode_input_data(input_data):
    """
    Apply label encoding to categorical features.
    If a value is not in the encoder's classes_, encode as -1.
    Special handling for gender mapping.
    """
    encoded_data = input_data.copy()

    # Handle gender mapping first
    if 'gioitinh' in input_data:
        gender_value = str(input_data['gioitinh']).strip()
        gender_mappings = CONFIG.get('field_mappings', {}).get('gioitinh', {})
        encoded_data['gioitinh'] = gender_mappings.get(gender_value, gender_mappings.get('default', -1))

    # Handle other categorical features
    for feature in CONFIG['categorical_features']:
        if feature == 'gioitinh':
            continue
        if feature in input_data:
            value = input_data[feature]
            if feature in label_encoders:
                encoder = label_encoders[feature]
                # Check if value exists in encoder's classes
                if value in encoder.classes_:
                    encoded_value = encoder.transform([value])[0]
                    encoded_data[feature] = convert_np_types(encoded_value)
                else:
                    encoded_data[feature] = CONFIG['default_values']['unknown_category']
            else:
                logger.warning(f"No encoder found for feature: {feature}")
                encoded_data[feature] = CONFIG['default_values']['unknown_category']

    # Convert sotc, hocky, namhoc, khoahoc, dtbhk1-22, sotchk1-22 to numeric types
    numeric_features = ['sotc', 'hocky', 'namhoc', 'khoahoc']
    for i in range(1, 23):
        numeric_features.append(f'dtbhk{i}')
        numeric_features.append(f'sotchk{i}')

    for feature in numeric_features:
        if feature in encoded_data:
            try:
                encoded_data[feature] = float(encoded_data[feature])
            except (ValueError, TypeError):
                # Use a numeric default value, e.g., -1
                encoded_data[feature] = CONFIG['default_values'].get('numeric_default', -1)

    # Convert any remaining numpy types
    for key in encoded_data:
        encoded_data[key] = convert_np_types(encoded_data[key])

    return encoded_data

# Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'danger')
            return redirect(url_for('signup'))

        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash('Username is already taken. Please choose another one.', 'danger')
            return redirect(url_for('signup'))

        # Create a new user
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            # Clear the session to prevent data leakage between users
            session.clear()
            
            # Log in the user
            login_user(user)
            session['user_id'] = user.id  # Optional: store user ID in the session
            
            # Ensure session changes are saved
            session.modified = True
            
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/prepare_preview', methods=['POST'])
@login_required
def prepare_preview():
    """Prepare input data preview before prediction"""
    try:
        # Retrieve student and course data with debug logging
        student_input = session.get('student_data')
        course_input = session.get('selected_course')

        logger.info(f"Student data from session: {student_input}")
        logger.info(f"Course data from session: {course_input}")

        if not student_input:
            return jsonify({'error': 'Please upload student information first'}), 400
        if not course_input:
            return jsonify({'error': 'Please select a course first'}), 400

        # Filter out -1 values from student data for dtbhk and sotchk
        filtered_student_data = {}
        for key, value in student_input.items():
            if key.startswith(('dtbhk', 'sotchk')):
                if value != -1:
                    filtered_student_data[key] = value
            else:
                filtered_student_data[key] = value

        # Combine filtered student and course data for preview
        preview_data = []

        # Add student information
        if filtered_student_data:
            preview_data.append({'feature': 'Student Information', 'value': '', 'isHeader': True})
            for key, value in filtered_student_data.items():
                if value not in [-1, '', 'nan', None]:
                    display_key = CONFIG.get('display_labels', {}).get(key, key)
                    preview_data.append({'feature': display_key, 'value': value})

        # Add course information
        if course_input:
            preview_data.append({'feature': 'Course Information', 'value': '', 'isHeader': True})
            for key, value in course_input.items():
                if value not in [-1, '', 'nan', None]:
                    display_key = CONFIG.get('display_labels', {}).get(key, key)
                    preview_data.append({'feature': display_key, 'value': value})

        # -----------------------------
        # NEW LOGIC: Encode the combined data
        # -----------------------------
        # Merge filtered student data and course data into one input dictionary
        combined_input = {**filtered_student_data, **course_input}

        # Encode this combined input using your encode_input_data function
        encoded_input = encode_input_data(combined_input)

        # Store both the preview and encoded input in the session
        session['input_preview'] = {
            'student_data': filtered_student_data,
            'course_data': course_input
        }
        session['encoded_input'] = encoded_input  # <-- CRUCIAL FOR PREDICTION

        logger.info(f"Preview data prepared: {preview_data}")
        logger.info(f"Encoded input data: {encoded_input}")

        return jsonify({
            'success': True,
            'preview_data': preview_data
        })
    except Exception as e:
        logger.exception("Error preparing preview data")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    # Separate session variables for student and course data
    timetable_data = session.get('timetable_data', [])  # Change default to empty list
    student_data = session.get('student_data', {})
    selected_course = session.get('selected_course', {})
    selected_model_name = session.get('selected_model_name', None)
    predictions_table = session.get('predictions_table', None)
    input_preview = {}  # New variable for input preview

    if request.method == 'POST':
        logger.info("Received POST request in dashboard.")
        logger.info(f"Form data: {request.form}")
        logger.info(f"Files data: {request.files}")
        
        if 'action' in request.form:
            action = request.form['action']
            logger.info(f"Action received: {action}")
            if action == 'process_selected':
                # Handle selected course data
                selected_data = {
                    'mamh': request.form.get('mamh'),
                    'sotc': request.form.get('sotc'),
                    'hocky': request.form.get('hocky'),
                    'namhoc': request.form.get('namhoc'),
                    'khoahoc': request.form.get('khoahoc'),
                    'hedt': request.form.get('hedt'),
                    'khoa': request.form.get('khoa')
                }
                logger.info(f"Selected course data: {selected_data}")
                session['selected_course'] = selected_data
                flash('Course data selected successfully!', 'success')
                logger.info("Course data saved to session.")
                return redirect(url_for('dashboard'))

            elif action == 'predict':
                try:
                    # Use the previously prepared encoded input data
                    input_data = session.get('encoded_input', {})
                    logger.info(f"Encoded input data for prediction: {input_data}")
                    if not input_data:
                        flash('No input data available. Please try again.', 'danger')
                        logger.warning("No encoded input data found in session.")
                        return redirect(url_for('dashboard'))

                    selected_model = request.form.get('model')
                    if selected_model not in MODELS:
                        flash('Selected model is invalid.', 'danger')
                        logger.error(f"Invalid model selected: {selected_model}")
                        return redirect(url_for('dashboard'))

                    model_path = MODELS[selected_model]
                    loaded_model = load_model(model_path)
                    logger.info(f"Loaded model from path: {model_path}")

                    # Create ordered_data ensuring all EXPECTED_FEATURES are present
                    ordered_data = {
                        field: input_data.get(field, CONFIG['default_values'].get('numeric_default', -1)) 
                        for field in EXPECTED_FEATURES
                    }

                    input_df = pd.DataFrame([ordered_data])
                    logger.info('Raw input data for prediction:')
                    logger.info(input_df)

                    # Ensure the DataFrame columns match EXPECTED_FEATURES in order
                    input_df = input_df[EXPECTED_FEATURES]

                    logger.info(f"Input DataFrame columns: {input_df.columns.tolist()}")

                    # Verify data types before prediction
                    logger.info("Verifying data types before prediction:")
                    for column in input_df.columns:
                        logger.info(f"{column}: {input_df[column].dtype}")

                    # Perform prediction
                    predictions = loaded_model.predict(input_df).astype(np.float64)
                    prediction_value = predictions[0] if predictions.size == 1 else predictions.tolist()
                    logger.info(f"Predictions: {prediction_value}")

                    # Retrieve selected course data
                    selected_course = session.get('selected_course', {})

                    # Combine predictions with selected course fields
                    combined_data = selected_course.copy()
                    combined_data['Prediction'] = prediction_value
                    
                    combined_data_df = pd.DataFrame([combined_data])
                    # combined_data_df.drop(combined_data_df.columns[0], axis=1, inplace=True)
                    combined_data_df.reset_index(drop=True, inplace=True)
                    combined_data_df.rename(columns={
                        'mamh': 'Course Code',
                        'sotc': 'Credits',
                        'hocky': 'Semester',
                        'namhoc': 'Year',
                        'khoahoc': 'Academic Year',
                        'hedt': 'Education Program',
                        'khoa': 'Faculty',
                        'Prediction': 'Predicted Score'
                    }, inplace=True)
                    # Create predictions_table with course fields and prediction
                    predictions_table = combined_data_df.to_html(classes='table table-striped', index=False, justify='center')

                    logger.info("Generated predictions table with course fields.")

                    # Store predictions in session with proper conversion
                    session['predictions'] = prediction_value
                    session['predictions_table'] = predictions_table
                    session['selected_model_name'] = selected_model

                    logger.info(f"Predictions stored in session: {session['predictions']}")

                    flash('Predictions made successfully!', 'success')
                except Exception as e:
                    logger.exception("Prediction failed.")
                    flash(f'Prediction failed: {str(e)}', 'danger')
                    return redirect(url_for('dashboard'))

                return redirect(url_for('dashboard'))

        # File upload handling
        if 'file' not in request.files:
            flash('No file part in the request.', 'error')
            logger.error("No file part in the request.")
            return redirect(request.url)

        file = request.files['file']
        # Check if user submitted empty form
        if file.filename == '':
            flash('No file selected.', 'error')
            logger.error("No file selected for upload.")
            return redirect(request.url)

        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving uploaded file to: {file_path}")
            file.save(file_path)
            logger.info("File saved successfully.")

            if 'upload_timetable' in request.form:
                timetable_data = process_timetable(file_path)  # Now returns list of dicts
                session['timetable_data'] = timetable_data
                logger.info("Timetable data saved to session.")
                flash('Timetable uploaded and processed!', 'success')

            elif 'upload_student_info' in request.form:
                student_data = process_student_info(file_path)
                session['student_data'] = student_data
                logger.info("Student data saved to session.")
                flash('Student info uploaded and processed!', 'success')
            else:
                flash('Invalid form submission.', 'error')
                logger.error("Invalid form submission.")
    # Initialize default student data if not present
    if not student_data:
        student_data = {
            'gioitinh': -1,
            'lopsh': '',
            'namsinh': -1,
            'noisinh': 'nan',
            'hedt': '',
            'khoa': '',
            'mamh': -1,
            'malop': -1,
            'sotc': -1,
            'hocky': -1,
            'namhoc': -1,
            'khoahoc': -1,
            'trangthai': -1,
            'tinhtrang': -1,
            'mamh_tt': -1,
            'namhoc_monhoc': -1,
            'hocky_monhoc': -1,
            'gap_hocky': -1,
            'hocky_monhoc_count': -1,
            'namhoc_monhoc_count': -1,
            # dtbhk1 to dtbhk22 and sotchk1 to sotchk22 initialized to -1
        }
        for i in range(1, 23):
            student_data[f'dtbhk{i}'] = -1
            student_data[f'sotchk{i}'] = -1
        session['student_data'] = student_data

        ordered_student_data = {
            field: student_data.get(field, CONFIG['default_values'].get('numeric_default', -1)) 
            for field in CONFIG['feature_order']
        }
        session['student_data'] = ordered_student_data
        student_data = ordered_student_data
        logger.info("Initialized default student data in session.")

    # Retrieve input_preview from session
    input_preview = session.get('input_preview', {})
    logger.info(f"Input preview retrieved from session: {input_preview}")

    return render_template('dashboard.html', 
                           timetable_data=timetable_data,
                           student_data=student_data,
                           selected_course=selected_course,
                           selected_model_name=selected_model_name,
                           predictions_table=predictions_table,
                           input_preview=input_preview,
                           models=list(MODELS.keys()),
                           config=CONFIG,
                           places_of_birth=CONFIG['places_of_birth'])

@app.route('/download/<file_type>')
@login_required
def download_file(file_type):
    data = []
    if file_type == 'timetable':
        data = session.get('timetable_data', [])
    elif file_type == 'student_info':
        data = [session.get('student_data', {})]
    elif file_type == 'predictions':
        data = session.get('predictions_table', None)
        if data:
            # Convert HTML table back to DataFrame if necessary
            # This assumes predictions are stored as HTML. Adjust as needed.
            df = pd.read_html(data)[0]
        else:
            df = pd.DataFrame([])
    else:
        flash('Invalid file type.', 'error')
        return redirect(url_for('dashboard'))

    if not data:
        flash('No data found to download.', 'error')
        return redirect(url_for('dashboard'))

    if file_type == 'predictions' and not df.empty:
        csv_data = df.to_csv(index=False)
    else:
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
    
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = f"attachment; filename={file_type}.csv"
    response.headers["Content-Type"] = "text/csv"
    return response

@app.route('/update_student_info', methods=['POST'])
@login_required
def update_student_info():
    """Update student info in session"""
    data = request.get_json()
    
    # Get current student data
    student_data = session.get('student_data', {})
    
    # Update noisinh
    if 'noisinh' in data:
        student_data['noisinh'] = data['noisinh']
        # Clear any cached input preview to ensure it's regenerated with new noisinh
        session.pop('input_preview', None)
        session.pop('encoded_input', None)
    
    # Save back to session
    session['student_data'] = student_data
    session.modified = True
    
    return jsonify({
        'success': True,
        'updated_value': data.get('noisinh')
    })

@app.route('/suggestion', methods=['POST'])
@login_required
def suggestion():
    """Get suggestion using Gemini model"""
    # Get predictions from session
    predictions = session.get('predictions', [])
    if not predictions:
        logger.warning("No predictions found in session.")
        return jsonify({'error': 'Please make predictions first before getting suggestions'}), 400

    input_preview = session.get('input_preview', {})
    if not input_preview:
        logger.warning("No input_preview found in session.")
        return jsonify({'error': 'No input preview available'}), 400

    # Log the retrieved predictions
    logger.info(f"Retrieved predictions from session: {predictions}")
    if (predictions < 0 or predictions > 10):
        return jsonify({'error': 'Invalid predictions'}), 400
    # Format input data for Gemini
    preview_text = "Student Data Analysis:\n"
    student_data = input_preview.get('student_data', {})
    for key, value in student_data.items():
        if key.startswith(('dtbhk', 'sotchk')) and value != -1:
            preview_text += f"{key}: {value}\n"
    print(preview_text)

    predictions_str = str(predictions)

    prompt = f"""
    Phân tích kết quả dự đoán điểm môn học dựa trên dữ liệu sau:
    
    {preview_text}
    với sotchk: số tín chỉ trong các kỳ, khi trả lời cần ghi rõ là số tín chỉ trong kì, đừng để sotchk
    và dtbhk: điểm trung bình trong kỳ đó, khi trả lời cần ghi rõ điểm trung bình trong kì, đừng để dtbhk
    
    kết quả dự đoán: {predictions_str}
    
    Hãy phân tích theo format sau:
    
    Kết quả dự đoán và đánh giá sơ bộ:
    - Điểm dự đoán là bao nhiêu
    - So sánh với điểm trung bình các kỳ trước
    - Đánh giá mức độ (khá, giỏi, trung bình...)
    
    PHÂN TÍCH CHI TIẾT:
    
    1. Xu hướng điểm số:
    - Phân tích các điểm số qua từng kỳ
    - Nhận xét về sự biến động
    
    2. Ảnh hưởng của các học kỳ gần nhất:
    - Phân tích 3-4 kỳ gần nhất
    - Tác động đến kết quả dự đoán
    
    3. Yếu tố tín chỉ:
    - Phân tích số tín chỉ các kỳ
    - Ảnh hưởng đến kết quả dự đoán
    
    4. Kết luận và khuyến nghị:
    - Nhận xét tổng quan
    - Đề xuất cho sinh viên
    """

    logger.info("Preparing to get suggestion from Gemini API.")
    try:
        response_text = get_suggestion(prompt)
        logger.info("Received suggestion from Gemini API.")

        # Split into suggestion and explanation
        parts = response_text.split('\n\n', 1)
        suggestion_text = parts[0].strip()
        explanation_text = parts[1] if len(parts) > 1 else "Analysis based on your academic performance trends."
        
        # Format the explanation text for better readability
        explanation_text = explanation_text.replace('*', '').strip()
        # print raw explanation text
        print(explanation_text)
        return jsonify({
            # 'suggestion': suggestion_text,
            'explanation': explanation_text
        })
    except Exception as e:
        logger.error("Gemini API call failed:", exc_info=True)
        return jsonify({
            'suggestion': "Unable to generate suggestion",
            'explanation': str(e)
        }), 500

@app.route('/remove_data', methods=['POST'])
@login_required
def remove_data():
    """Remove student or course data from session"""
    data = request.get_json()
    remove_type = data.get('type')
    
    if remove_type == 'student':
        # Clear all data since student info is required first
        session.pop('student_data', None)
        session.pop('timetable_data', None)
        session.pop('selected_course', None)
        session.pop('predictions_table', None)
        session.pop('predictions', None)
        session.pop('input_preview', None)
        session.pop('encoded_input', None)
        flash('Student information has been removed.', 'success')
    elif remove_type == 'course':
        # Only clear course-related data
        session.pop('selected_course', None)
        session.pop('predictions_table', None)
        session.pop('predictions', None)
        session.pop('input_preview', None)
        session.pop('encoded_input', None)
        flash('Course selection has been removed.', 'success')
    
    return jsonify({'success': True})

if __name__ == '__main__':
    init_db() 
    app.run(debug=True)