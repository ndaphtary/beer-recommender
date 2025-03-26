import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
df = pd.read_csv('beer_profile_and_ratings.csv')
df = df[['Beer Name (Full)','ABV', 'Style', 'Astringency', 'Body', 'Alcohol', 'Bitter',
         'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty']]
df['Style'] = df['Style'].fillna('Unknown')
df[df.columns[2:]] = df[df.columns[2:]].fillna(0)

# Streamlit UI
st.set_page_config(page_title="Beer Recommender", layout="wide")
st.title("Beer Flavor Recommender")
st.markdown("Adjust the sliders to set your preferred flavor profile. We'll suggest beers that match your taste!")

# Get user flavor preferences with sliders
flavor_weights = {}
st.sidebar.header("Set Your Flavor Preferences")
for attr in df.columns[3:]:
    flavor_weights[attr] = st.sidebar.slider(attr, 0.0, 1.0, 0.5, 0.1)

# Normalize and compute similarity
scaler = MinMaxScaler()
flavor_attributes = df[df.columns[3:]]
scaled_flavor_attributes = scaler.fit_transform(flavor_attributes)

user_vector = np.array([flavor_weights[attr] for attr in flavor_attributes.columns]).reshape(1, -1)
scaled_user_vector = user_vector

similarities = cosine_similarity(scaled_flavor_attributes, scaled_user_vector)
df['Similarity'] = similarities

# Top recommendations
top_recs = df.sort_values('Similarity', ascending=False).drop_duplicates(['Beer Name (Full)']).head(10)

st.subheader("Top 10 Recommended Beers for You")
st.dataframe(top_recs[['Beer Name (Full)', 'Style','ABV', 'Similarity']].reset_index(drop=True))

st.caption("Powered by your taste!")
