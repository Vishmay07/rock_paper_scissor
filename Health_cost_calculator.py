# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Cell 2: Load the dataset
# Download the data if not already available or load it from a path
book_file = '/content/Book-Crossing/ratings.csv'
books_file = '/content/Book-Crossing/books.csv'

# Load ratings and books
ratings_df = pd.read_csv(book_file, names=["user_id", "book_id", "rating"], header=0)
books_df = pd.read_csv(books_file, names=["book_id", "title"], header=0)

# Cell 3: Data Preprocessing and Filtering
# Filter out books with less than 100 ratings and users with less than 200 ratings

# 1. Filter out books with less than 100 ratings
book_counts = ratings_df['book_id'].value_counts()
books_filtered = book_counts[book_counts >= 100].index
ratings_filtered = ratings_df[ratings_df['book_id'].isin(books_filtered)]

# 2. Filter out users with less than 200 ratings
user_counts = ratings_filtered['user_id'].value_counts()
users_filtered = user_counts[user_counts >= 200].index
ratings_filtered = ratings_filtered[ratings_filtered['user_id'].isin(users_filtered)]

# Create the user-item matrix
user_item_matrix = ratings_filtered.pivot_table(index='user_id', columns='book_id', values='rating')

# Fill missing values with 0 (indicating no rating)
user_item_matrix = user_item_matrix.fillna(0)

# Cell 4: Train K-Nearest Neighbors model
# We use NearestNeighbors to find books similar to a given book
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
knn.fit(user_item_matrix.T)  # Transpose to match book-based similarity

# Cell 5: Recommendation Function
def get_recommends(book_title: str):
    # Find the book_id for the title
    book_id = books_df[books_df['title'] == book_title]['book_id'].values
    if len(book_id) == 0:
        return f"Book '{book_title}' not found in the dataset."
    
    book_id = book_id[0]
    
    # Find the book's neighbors using the KNN model
    distances, indices = knn.kneighbors(user_item_matrix[book_id].reshape(1, -1), n_neighbors=6)
    
    # Prepare the recommendations: list of book titles and their distances
    recommendations = []
    for i in range(1, len(indices[0])):
        recommended_book_id = user_item_matrix.columns[indices[0][i]]
        recommended_book_title = books_df[books_df['book_id'] == recommended_book_id]['title'].values[0]
        recommendations.append([recommended_book_title, distances[0][i]])
    
    return [book_title, recommendations]

# Test the recommendation function
recommendations = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(recommendations)
