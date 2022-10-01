from select import select
from turtle import title
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data Collection and Pre-Processing
# loading data from the CSV file -> Pandas data frame

movies_data = pd.read_csv('/Users/mirajpatel/Downloads/movie_recommender/movies.csv')
movies_data.head()

# choose the number of rows and columns in data frame

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# replace any missing values from the data frame with a null string

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# combining our selected features

combined_features = movies_data['genres'] + ' '+ movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# converting textual data into feature vectors so they turn into numerical data

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

# getting the similarity confidence value by using cosine similarity

similarity = cosine_similarity(feature_vectors)

# getting movie from users

movie_name = input("Enter your favorite movie: ")

# creating a list with all the movie names given in the dataset

titles_list = movies_data['title'].tolist()

# finding the close match for the movie name given by the user

close_match = difflib.get_close_matches(movie_name, titles_list)

close_match = close_match[0]

# finding the index of the movie

index = movies_data[movies_data.title == close_match]['index'].values[0]
print(index)

# getting a list of similar movies

similarity_scores = list(enumerate(similarity[index]))

sorted_scores = sorted(similarity_scores, key = lambda x:x[1], reverse=True)
# print the name of the similar movies based on the index

print('Movies suggested for you: \n')

i = 1
for movie in sorted_scores:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if (i < 21):
        print(i, ".", title_from_index)
        i += 1