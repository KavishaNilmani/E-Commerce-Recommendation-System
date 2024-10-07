from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# load files===========================================================================================================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# database configuration---------------------------------------
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
#app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecom"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root@localhost:3307/ecom"


app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

# def content_based_recommendations(train_data, item_name, top_n=10):
#     # Check if the item name exists in the training data
#     if item_name not in train_data['Name'].values:
#         print(f"Item '{item_name}' not found in the training data.")
#         return pd.DataFrame()
#
#     # Create a TF-IDF vectorizer for item descriptions
#     tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#
#     # Apply TF-IDF vectorization to item descriptions
#     tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
#
#     # Calculate cosine similarity between items based on descriptions
#     cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
#
#     # Find the index of the item
#     item_index = train_data[train_data['Name'] == item_name].index[0]
#
#     # Get the cosine similarity scores for the item
#     similar_items = list(enumerate(cosine_similarities_content[item_index]))
#
#     # Sort similar items by similarity score in descending order
#     similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
#
#     # Get the top N most similar items (excluding the item itself)
#     top_similar_items = similar_items[1:top_n+1]
#
#     # Get the indices of the top similar items
#     recommended_item_indices = [x[0] for x in top_similar_items]
#
#     # Get the details of the top similar items
#     recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
#
#     return recommended_items_details


# routes===============================================================================
# List of predefined image URLs

# Collaborative Filtering Function
def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    # Check if the user ID exists in the DataFrame
    if target_user_id not in train_data['ID'].values:
        print(f"User ID {target_user_id} not found in dataset.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Check if the user has rated any products
    user_ratings = train_data[train_data['ID'] == target_user_id]
    if user_ratings.empty:
        print(f"User ID {target_user_id} has not rated any products.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Create the user-item matrix
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)

    # Get the index of the target user
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]

    # Sort users by similarity
    similar_users_indices = user_similarities.argsort()[::-1][1:]
    recommended_items = []

    for user_index in similar_users_indices:
        # Get items rated by the similar user but not by the target user
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    # Get the details of recommended items
    recommended_items_details = train_data[train_data['ProdID'].isin(recommended_items)][
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
    return recommended_items_details.head(10)


def content_based_recommendations(train_data, item_name, top_n=10, similarity_threshold=0.2):
    # Use case-insensitive matching for partial matches in 'Name', 'Brand', or 'Tags'
    matches = train_data[
        train_data['Name'].str.contains(item_name, case=False, na=False) |
        train_data['Brand'].str.contains(item_name, case=False, na=False) |
        train_data['Tags'].str.contains(item_name, case=False, na=False)
    ]

    if matches.empty:
        print(f"No matches found for '{item_name}'.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Get the indices of matched items
    matched_indices = matches.index.tolist()

    recommended_items = []

    # Collect recommendations based on those matched indices
    for item_index in matched_indices:
        similar_items = list(enumerate(cosine_similarities_content[item_index]))
        similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

        # Add items to recommended list but limit to top_n and apply similarity threshold
        for sim_index, similarity_score in similar_items[1:top_n + 1]:  # Exclude the item itself
            if similarity_score > similarity_threshold:  # Only include items above the threshold
                recommended_items.append((sim_index, similarity_score))
            if len(recommended_items) >= top_n:  # Stop if we reach the limit
                break

    # Get unique recommended item indices and sort by highest similarity
    recommended_items = sorted(list(set(recommended_items)), key=lambda x: x[1], reverse=True)[:top_n]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in recommended_items]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[recommended_item_indices][
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    # Filter out items without image URLs
    recommended_items_details = recommended_items_details[recommended_items_details['ImageURL'].notna()]

    return recommended_items_details



random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    # "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
    "static/img/img_9.png",
    "static/img/img_10.png",
    "static/img/img_11.png",

]


# @app.route("/")
# def index():
#     # Create a list of random image URLs for each product
#     random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
#     price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
#     return render_template('index.html', trending_products=trending_products.head(8),truncate=truncate,
#                            random_product_image_urls=random_product_image_urls,random_price=random.choice(price))
@app.route("/")
def index():
    # Shuffle and select unique random image URLs for each product
    unique_random_product_image_urls = random.sample(random_image_urls,
                                                     k=min(len(trending_products), len(random_image_urls)))

    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

    return render_template('index.html',
                           trending_products=trending_products.head(8),
                           truncate=truncate,
                           random_product_image_urls=unique_random_product_image_urls,
                           random_price=random.choice(price))


#routes
# @app.route("/main")
# def main():
#     return render_template('main.html')
#
@app.route("/main")
def main():
    # Pass an empty DataFrame to avoid errors when no recommendation is made
    empty_recommendations = pd.DataFrame()
    return render_template('main.html', content_based_rec=empty_recommendations)


#routes
@app.route("/index")
def indexredirect():
    return render_template('index.html')

# @app.route("/signup", methods=['POST','GET'])
# def signup():
#     if request.method=='POST':
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
#
#         new_signup = Signup(username=username, email=email, password=password)
#         db.session.add(new_signup)
#         db.session.commit()
#
#         # Create a list of random image URLs for each product
#         random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
#         price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
#         return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
#                                random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
#                                signup_message='User signed up successfully!'
#                                )
# Route for signup page

@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Add new user to the Signup table
        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed up successfully!')

@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']

        # Check if the user exists in the Signup table
        user = Signup.query.filter_by(username=username, password=password).first()
        if user:
            # Store the username in the session
            session['username'] = user.username

            # Create a list of random image URLs for each product
            unique_random_product_image_urls = random.sample(random_image_urls,
                                                             k=min(len(trending_products), len(random_image_urls)))

            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

            return render_template('index.html',
                                   trending_products=trending_products.head(8),
                                   truncate=truncate,
                                   random_product_image_urls=unique_random_product_image_urls,
                                   random_price=random.choice(price),
                                   signup_message=f'Welcome! {user.username},👋')
        else:
            return render_template('signin.html', error='Invalid username or password')
# @app.route("/recommendations", methods=['POST', 'GET'])
# def recommendations():
#     if request.method == 'POST':
#         prod = request.form.get('prod')
#         nbr = int(request.form.get('nbr'))
#         content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)
#
#         if content_based_rec.empty:
#             message = "No recommendations available for this product."
#             return render_template('main.html', message=message, content_based_rec=pd.DataFrame())
#         else:
#             # Create a list of random image URLs for each recommended product
#             random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
#             price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
#             return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
#                                    random_product_image_urls=random_product_image_urls,
#                                    random_price=random.choice(price))
@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this search."
            return render_template('main.html', message=message, content_based_rec=pd.DataFrame())
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = []
            for _ in range(len(content_based_rec)):
                image_url = random.choice(random_image_urls)
                random_product_image_urls.append(image_url)

            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price))

# New route for collaborative recommendations
@app.route('/collaborative', methods=['GET', 'POST'])
def collaborative():
    recommendations = None
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        print(f"User ID entered: {user_id}")  # Debug output

        # Ensure the user_id is in the correct format
        try:
            user_id = int(user_id)  # Convert to integer if necessary
        except ValueError:
            print("Invalid User ID format.")
            recommendations = pd.DataFrame()  # Return an empty DataFrame
            return render_template('collab.html', recommendations=recommendations)

        # Call the collaborative filtering function
        recommendations = collaborative_filtering_recommendations(train_data, user_id)

        print(f"Recommendations: {recommendations}")  # Debug output

    return render_template('collab.html', recommendations=recommendations)


if __name__=='__main__':
    app.run(port=5000, debug=True)


