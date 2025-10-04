import pandas as pd
import os
import sys
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
data_dir = r"C:\Users\sudha\Desktop\dhanush\Personal DS\Computer Vision\Fashion recommendation system\processed_data"
model = load_model(
    r"C:\Users\sudha\Desktop\dhanush\Personal DS\Computer Vision\Fashion recommendation system\ImgFeature_model.h5"
)

all_df = pd.read_pickle(os.path.join(data_dir,"all_img_features.pkl"))

def calc_distance(img_array):
    # Extract feature vector for query image
    new_feature = model.predict(img_array)   # shape (1, 2048)
    new_feature = new_feature.reshape((1, -1))
    all_features = np.vstack(all_df["feature_arr"].values) 
     # Compute cosine similarity
    sims = cosine_similarity(new_feature, all_features)[0]  # shape (N,)
    
    all_df["curr_sim"] = sims
    
    return all_df
    
def get_input(img_path):
    img = image.load_img(img_path,target_size=(224,224,3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)
    return img_array

def recommendation_pipeline(img_file_path,top_k=5):
    
    #filter_cols = filters_json.keys()
    # get input
    img_array = get_input(img_file_path)
    # get custom df
    custom_df = calc_distance(img_array)
    # perform filters
    """
    for col in filter_cols:
        subsets = filters_json[col] # list
        custom_df = custom_df[custom_df[col].isin(subsets)] 
    """
    custom_df_sorted = custom_df.sort_values("curr_sim",ascending=False).head(top_k)
    print("............done with calcs............")
    #images_list = custom_df_sorted["img_file"].tolist() 
    #img_paths = [] 
    #for image_file in images_list:
    #    image_file = image_file+".jpg"
    #    image_dir = r"C:\Users\sudha\Desktop\dhanush\Personal DS\Computer Vision\Fashion recommendation system\archive (8)\images"
    #    img_paths.append(os.path.join(image_dir,image_file))
    return custom_df_sorted
