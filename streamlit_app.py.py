import streamlit as st
from PIL import Image
import pandas as pd
import os
from RecommendationSystem import recommendation_pipeline

image_dir = r"C:\Users\sudha\Desktop\dhanush\Personal DS\Computer Vision\Fashion recommendation system\archive (8)\images"
data_dir = r"C:\Users\sudha\Desktop\dhanush\Personal DS\Computer Vision\Fashion recommendation system\processed_data"
all_df = pd.read_pickle(os.path.join(data_dir,"all_img_features.pkl"))
styles_df = pd.read_csv(r"C:\Users\sudha\Desktop\dhanush\Personal DS\Computer Vision\Fashion recommendation system\archive (8)\styles.csv",encoding="utf-8",on_bad_lines="skip")
st.set_page_config(layout="wide")
st.title("Fashion Recommendation System")

st.write("### Select an image from catalog")


# Show small gallery of images (first 20 for demo)
gallery_df = all_df.sample(30,random_state=42)   # you can paginate later
cols = st.columns(5)

selected_idx = st.session_state.get("selected_idx", None)

for i, row in enumerate(gallery_df.itertuples()):
    with cols[i % 5]:
        image_path = os.path.join(image_dir,row.img_file+".jpg")
        img = Image.open(image_path)
        if st.button(f"Select {row.img_file}", key=f"btn_{i}"):
            selected_idx = row.Index
            st.session_state["selected_idx"] = selected_idx
        st.image(img, caption=f"{row.img_file}", use_container_width=True)

# ------------------------------
# Show recommendations
# ------------------------------
if selected_idx is not None:
    st.markdown("---")
    st.write("### 🔍 Similar Items")

    selected_row = all_df.iloc[selected_idx]
    image_path = os.path.join(image_dir,selected_row.img_file+".jpg")
    selected_img = Image.open(image_path)
    st.image(selected_img, caption="You selected", width=200)

    # Get recommendations
    results = recommendation_pipeline(image_path, top_k=6)
    results = pd.merge(styles_df,results,left_on="id",right_on="img_file")
#"gender	masterCategory	subCategory	articleType	baseColour	season	year	usage	productDisplayName
#"
    # Display recommended images with tags
    rec_cols = st.columns(3)
    for j, row in enumerate(results.itertuples()):
        with rec_cols[j % 3]:
            try:
                image_path = os.path.join(image_dir,row.img_file+".jpg")
                rec_img = Image.open(image_path)
                st.image(rec_img, use_container_width=True)
                st.write(f"**Tags:** {row.gender}, {row.masterCategory},{row.subCategory},{row.articleType},{row.baseColour},{row.season},{row.year},{row.usage}")
                st.caption(f"Name: {row.productDisplayName} \nSimilarity: {row.curr_distance:.2f}")
            except:
                st.write(row.img_file)
