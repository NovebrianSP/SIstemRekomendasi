import pandas as pd
from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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

# Preprocess text for description similarity
def preprocess_text(text):
    """Preprocess text by tokenizing, lowercasing, and removing common stopwords."""
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stop_words = set(stopwords.words('indonesian'))  # Stopwords for Indonesian
    stemmer = PorterStemmer()

    tokens = text.lower().split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return set(tokens)

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

    def calculate_combined_similarity(row):
        row_desc_set = preprocess_text(row['Description'])
        desc_similarity = len(user_desc_set & row_desc_set) / len(user_desc_set | row_desc_set) if len(user_desc_set | row_desc_set) != 0 else 0
        category_similarity = 1 if row['Category'] == category else 0
        return 0.8 * desc_similarity + 0.2 * category_similarity

    filtered_df['Similarity'] = filtered_df.apply(calculate_combined_similarity, axis=1)

    # Compute final score by combining similarity, rating, and price
    filtered_df['Final_Score'] = (
        0.7 * filtered_df['Similarity'] + 
        0.2 * (filtered_df['Rating'] / 5) + 
        0.1 * (1 - filtered_df['Price'] / filtered_df['Price'].max())
    )

    # Sort recommendations by Final_Score
    sorted_df = filtered_df.sort_values(by='Final_Score', ascending=False)
    recommendations = sorted_df[['Place_Name', 'Price', 'Rating', 'Similarity', 'Final_Score']].to_dict(orient='records')

    return render_template('result.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)