import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

hide_st_style = """
            <style>
            
            footer {visibility: hidden;}
            header{margin:0;}
            
            margin:0;
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

head_title = '<p style="font-family:sans-serif; color:White; font-size: 60px; text-align:center ;font-weight:bold;margin:0">FASHION GENIE</p>'
st.markdown(head_title, unsafe_allow_html=True)

sub_title = '<p style="font-family:sans-serif; color:White; font-size: 20px; text-align:center;">Genie For Your Fashion</p>'
st.markdown(sub_title, unsafe_allow_html=True)

#option menu
selected=option_menu(
    menu_title=None,
    options=["SEARCH"],
    icons=["camera"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)




#gif
st.markdown("")
def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_coding=load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_ecnepkno.json")
st_lottie(lottie_coding,height=290,key="coding",)



#search photos
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Search Using Photos')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
       
        upload_title = '<p style="font-family:sans-serif; color:White; font-size: 30px;font-weight:bold;margin-bottom:20px">UPLOADED IMAGE</p>'
        st.markdown(upload_title, unsafe_allow_html=True)
        display_image = Image.open(uploaded_file)
        st.image(display_image,width=180)
       
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        
        indices = recommend(features,feature_list)
       

        st.title("RECOMMENDATIONS")
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]],width=150)
        with col2:
            st.image(filenames[indices[0][1]],width=150)
        with col3:
            st.image(filenames[indices[0][2]],width=150)
        with col4:
            st.image(filenames[indices[0][3]],width=150)
        with col5:
            st.image(filenames[indices[0][4]],width=150)
    else:
        st.header("Some error occured in file upload")





