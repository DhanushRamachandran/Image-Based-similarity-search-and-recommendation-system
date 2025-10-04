import pandas as pd
import os
dir = r"C:\Users\sudha\Desktop\dhanush\Personal DS\Computer Vision\Fashion recommendation system\processed_data"
all_df = pd.DataFrame()
for file in os.listdir(dir):
    temp_df = pd.read_pickle(os.path.join(dir,file))
    all_df = pd.concat([all_df,temp_df],axis=0)
all_df.to_pickle(os.path.join(dir,"all_img_features.pkl"))