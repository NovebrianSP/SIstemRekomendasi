import os
import pandas as pd
from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset
file_path = 'E:/DATA/PythonProgram/SR/destinasi-wisata-indonesia.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse('Worksheet')

# Select relevant features
selected_features = df[['Place_Name', 'Category', 'City', 'Price', 'Rating', 'Time_Minutes', 'Description']]

# Handle missing values
selected_features['Time_Minutes'].fillna(selected_features['Time_Minutes'].median(), inplace=True)
selected_features['Description'].fillna("", inplace=True)

# Encode categorical variables
encoded_data = pd.get_dummies(selected_features, columns=['Category', 'City'], drop_first=True)

# Define features (X) and target (y)
X = encoded_data.drop(columns=['Place_Name', 'Description'])  # Drop Place_Name and Description as they are non-numerical
y = df['Category']  # Predicting the category of the place

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Preprocess text for description similarity using Jaccard Similarity
def preprocess_text(text):
    """Preprocess text by tokenizing and lowercasing."""
    return set(text.lower().split())

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    cities = df['City'].unique()
    categories = df['Category'].unique()
    return render_template('index.html', cities=cities, categories=categories)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form
    city = user_input.get('city')
    category = user_input.get('category')
    max_price = float(user_input.get('price'))
    description = user_input.get('description', "")

    # Filter the data based on user input
    filtered_df = df[(df['City'] == city) & (df['Category'] == category) & (df['Price'] <= max_price)]

    if filtered_df.empty:
        return render_template('result.html', recommendations=None)

    # Compute description similarity using Jaccard Similarity
    user_desc_set = preprocess_text(description)

    def calculate_jaccard_similarity(row):
        row_desc_set = preprocess_text(row['Description'])
        intersection = len(user_desc_set & row_desc_set)
        union = len(user_desc_set | row_desc_set)
        return intersection / union if union != 0 else 0

    filtered_df['Similarity'] = filtered_df.apply(calculate_jaccard_similarity, axis=1)

    # Sort recommendations by Similarity and Rating in descending order
    sorted_df = filtered_df.sort_values(by=['Similarity', 'Rating'], ascending=[False, False])
    recommendations = sorted_df[['Place_Name', 'Price', 'Rating', 'Similarity']].to_dict(orient='records')

    return render_template('result.html', recommendations=recommendations)

if __name__ == '__main__':
    # Ensure templates folder exists for Flask
    os.makedirs('templates', exist_ok=True)

    # Create templates for the app
    with open(os.path.join('templates', 'index.html'), 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Travel Recommendations</title>
        </head>
        <body>
            <h1>Travel Recommendation System</h1>
            <form action="/recommend" method="post">
                <label for="city">Select City:</label>
                <select name="city" id="city">
                    {% for city in cities %}
                    <option value="{{ city }}">{{ city }}</option>
                    {% endfor %}
                </select>
                <br>
                <label for="category">Select Category:</label>
                <select name="category" id="category">
                    {% for category in categories %}
                    <option value="{{ category }}">{{ category }}</option>
                    {% endfor %}
                </select>
                <br>
                <label for="price">Maximum Price:</label>
                <input type="number" name="price" id="price" step="0.01">
                <br>
                <label for="description">Describe Your Preference:</label>
                <textarea name="description" id="description" rows="4" cols="50"></textarea>
                <br>
                <button type="submit">Get Recommendations</button>
            </form>
        </body>
        </html>
        ''')

    with open(os.path.join('templates', 'result.html'), 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Recommendations</title>
        </head>
        <body>
            <h1>Recommendations</h1>
            {% if recommendations %}
                <ul>
                    {% for recommendation in recommendations %}
                    <li>
                        {{ recommendation.Place_Name }} - Price: {{ recommendation.Price }} - Rating: {{ recommendation.Rating }} - Similarity: {{ recommendation.Similarity|round(2) }}
                    </li>
                    {% endfor %}
                </ul>
            {% else %}
                <p>No recommendations found for the selected criteria.</p>
            {% endif %}
            <a href="/">Back to Home</a>
        </body>
        </html>
        ''')

    app.run(debug=True)