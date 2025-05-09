import streamlit as st
import pandas as pd
import pickle
import urllib.request
from PIL import Image

st.set_page_config(page_title="BookVerse Recommendation", page_icon="📚", layout="centered")

#CSS Styling 
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa, #948f81);
            background-attachment: fixed;
            padding: 2rem;
        }
        .main-title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #2c3e50;
            padding-bottom: 10px;
        }
        .description {
            text-align: center;
            font-size: 18px;
            color: #2c3e50;
            margin-bottom: 40px;
        }
        .recommend-card {
            transition: transform 0.3s ease;
            background-color: #ffffffcc;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            margin-bottom: 20px;
        }
        .recommend-card:hover {
            transform: scale(1.03);
        }
        .recommend-img {
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
        .recommend-title {
            font-size: 16px;
            font-weight: 600;
            color: #2c3e50;
            margin-top: 10px;
        }
        .recommend-title-section {
            text-align: center;
            font-weight: bold;
            font-size: 32px;
            color: #2c3e50;
            margin: 40px 0 30px;
            background-color: #ffffff33;
            padding: 12px 20px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .stButton > button {
            color: white;
            background-color: #2c3e50;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #1e1e1e;
        }
        label, .stSelectbox label, .stRadio label {
            color: #2c3e50 !important;
            font-weight: bold;
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📚 BookVerse </div>", unsafe_allow_html=True)
st.markdown("<div class='description'>Guiding you to unforgettable reads</div>", unsafe_allow_html=True)

# function to load images
def get_image(link):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
        request = urllib.request.Request(link, headers=headers)
        return Image.open(urllib.request.urlopen(request))
    except:
        return None

#load saved data from ipynb
@st.cache_data
def load_data():
    Matrix = pd.read_csv("books_processed2.csv", index_col=0)
    crosstab = pd.read_csv("crosstab.csv")
    with open("knn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return Matrix, crosstab, model

Matrix, crosstab, Model = load_data()


selected_book = st.selectbox(" Choose a book you’ve already read:", Matrix.index.tolist())


num_recommendations = st.radio(" How many recommendations would you like?", [4, 8, 12, 16, 20])

#here is a button to reccommend books
if st.button(" Recommend Books"):
    if selected_book:
        book_vector = Matrix.loc[selected_book].values.reshape(1, -1)
        _, indices = Model.kneighbors(book_vector, n_neighbors=num_recommendations + 1)

        st.markdown("<div class='recommend-title-section'>📚 Recommended Books</div>", unsafe_allow_html=True)

        recommended_books = []

        for idx in indices[0]:
            recommended_title = Matrix.index[idx]
            if recommended_title != selected_book:
                match = crosstab[crosstab['Book_Title'] == recommended_title]
                image = None
                if not match.empty:
                    image_url = match.iloc[0]['Image-URL-M']
                    image = get_image(image_url)

                recommended_books.append({
                    "title": recommended_title,
                    "image": image
                })

        #here is to check the screen about the cards 
        cols_per_row = 4
        for i in range(0, len(recommended_books), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, book in enumerate(recommended_books[i:i+cols_per_row]):
                with cols[j]:
                    st.markdown("<div class='recommend-card' style='text-align: center;'>", unsafe_allow_html=True)

                    if book['image']:
                        st.image(book['image'], use_container_width=True)
                    else:
                        st.write("Image not found.")

                    st.markdown(f"<div class='recommend-title'>{book['title']}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        #here is a button to recccommend again
        if st.button(" Recommend Again"):
            st.experimental_rerun()
