
from datetime import datetime
from flask import Flask, flash, get_flashed_messages, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import os
import tensorflow as tf
import re
from sklearn.preprocessing import LabelEncoder
import spacy

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")
# Use SQLite for simplicity
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)

# User model


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# Home route


@app.route('/')
def home():
    flashed_messages = get_flashed_messages(with_categories=True)
    error_message = next(
        (message for category, message in flashed_messages if category == 'error'), '')
    return render_template('index.html', error_message=error_message)

# Register route


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()

        if existing_user:
            # If the user already exists, flash a message and redirect to the login page
            flash('User already exists. Please login.', 'error')
            return redirect(url_for('home'))
        # Use pbkdf2:sha256 for hashing the password
        hashed_password = generate_password_hash(
            password, method='pbkdf2:sha256')

        # Create a new user object and add it to the database
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('home'))

    return render_template('register.html')

# Login route


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Query user from database
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            return render_template('home.html')
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('home'))

    return render_template('login.html')


model = tf.saved_model.load('skimlit_tribrid_model')
infer = model.signatures['serving_default'] # for prediction


class_names = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
label_encoder = LabelEncoder()
label_encoder.fit(class_names)


def split_chars(text):
    return " ".join(list(text))


def clean_text(text):
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_text(abstract_lines, max_sequence_length=1000, line_number_depth=15, total_lines_depth=20):
    """
    Preprocess the abstract lines into a format suitable for the model.

    Parameters:
    - abstract_lines: List of text lines from the abstract.
    - max_sequence_length: Maximum length for character sequences.
    - line_number_depth: Depth for one-hot encoding line numbers.
    - total_lines_depth: Depth for one-hot encoding total lines.

    Returns:
    - token_tensor: Tensor of text lines (not used in this example).
    - total_lines_one_hot: One-hot encoded total lines tensor.
    - line_numbers_one_hot: One-hot encoded line numbers tensor.
    - char_tensor: Tensor of character sequences.
    """
    total_lines_in_sample = len(abstract_lines)
    sample_lines = []

    # Prepare the sample lines dictionary
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)

    # Prepare one-hot encoding for line numbers
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    test_abstract_line_numbers_one_hot = tf.one_hot(
        test_abstract_line_numbers, depth=line_number_depth)

    # Prepare one-hot encoding for total lines
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    test_abstract_total_lines_one_hot = tf.one_hot(
        test_abstract_total_lines, depth=total_lines_depth)

    test_abstract_line_numbers_one_hot = tf.cast(
        test_abstract_line_numbers_one_hot, dtype=tf.int32)
    test_abstract_total_lines_one_hot = tf.cast(
        test_abstract_total_lines_one_hot, dtype=tf.int32)
    # Convert lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]
    abstract_chars = tf.expand_dims(abstract_chars, axis=-1)

    # Return prepared tensors
    return test_abstract_line_numbers_one_hot, test_abstract_total_lines_one_hot, tf.constant(abstract_lines), tf.constant(abstract_chars)


# def predict_text(text):
#     text_lines = text.splitlines()
#     token_inputs, total_lines_input, line_number_input, char_inputs = preprocess_text(
#         text_lines)
#     print(token_inputs.shape, total_lines_input.shape,
#           line_number_input.shape, char_inputs.shape)

#     # Load the model and prepare inputs
#     infer = model.signatures['serving_default']

#     inputs = {
#         'token_inputs': token_inputs,
#         'total_lines_input': total_lines_input,
#         'line_number_input': line_number_input,
#         'char_inputs': char_inputs
#     }

#     # Perform inference
#     predictions_probs = infer(**inputs)
#     predictions = tf.argmax(predictions_probs, axis=1)
#     predicted_classes = [class_names[i] for i in predictions.numpy()]

#     return predicted_classes


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'abstract_text' in request.form:
        abstract_text = request.form.get('abstract_text', '')

        # Use spaCy to split the abstract into sentences
        doc = nlp(abstract_text)
        abstract_lines = [str(sent)
                          for sent in doc.sents]  # Get sentences as strings
        print(abstract_lines)
        if not abstract_lines:
            return render_template('home.html', result=None, message="Please paste the abstract")

        # Process lines and prepare inputs for the model
        line_numbers_one_hot, total_lines_one_hot, abstract_lines, abstract_chars = preprocess_text(
            abstract_lines)

        try:
            # Print shapes of the tensors for debugging
            print("token_tensor shape:", abstract_lines.shape)
            print("total_lines_one_hot shape:", total_lines_one_hot.shape)
            print("line_numbers_one_hot shape:", line_numbers_one_hot.shape)
            print("char_tensor shape:", abstract_chars.shape)

            # Perform inference
            predictions = infer(line_number_input=line_numbers_one_hot,
                                total_lines_input=total_lines_one_hot,
                                token_inputs=abstract_lines,
                                char_inputs=abstract_chars)
            # Debug: print the predictions object
            print("Predictions:", predictions)

            # Extract and process the predictions
            pred_probs = predictions['output_layer']
            predicted_classes = tf.argmax(pred_probs, axis=-1).numpy()

            # Convert predictions to a readable format
            class_names = ['BACKGROUND', 'CONCLUSIONS',
                           'METHODS', 'OBJECTIVE', 'RESULTS']
            result = list(zip(abstract_lines.numpy(), [class_names[i]
                          for i in predicted_classes]))
            result = [(sentence.decode('utf-8'), classification) if isinstance(sentence, bytes)
                      else (sentence, classification) for sentence, classification in result]
            formatted_result = [(sentence, classification)
                                for sentence, classification in result]
            return render_template('home.html', result=formatted_result)
        except Exception as e:
            return f"An error occurred: {e}"

    return "No text provided"


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)  # Adjust according to your session management
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
