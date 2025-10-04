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
st.title("👗 Fashion Recommendation System")

st.write("### Select an image from catalog")


# Show small gallery of images (first 20 for demo)
gallery_df = all_df[:30]   # you can paginate later
cols = st.columns(5)

selected_idx = st.session_state.get("selected_idx", None)
print("selec idx iknit: ",selected_idx)
for i, row in enumerate(gallery_df.itertuples()):
    print(i)
    with cols[i % 5]:
        image_path = os.path.join(image_dir,row.img_file+".jpg")
        img = Image.open(image_path)
        if st.button(f"Select {row.img_file}", key=f"btn_{i}"):
            selected_idx = row.Index 
            print("selected idx: ",selected_idx) 
            st.session_state["selected_idx"] = selected_idx 
        st.image(img, caption=f"{row.img_file}", use_container_width=True)

# ------------------------------
# Show recommendations
# ------------------------------
if selected_idx is not None:
    st.markdown("---")
    st.write("### 🔍 Similar Items")

    selected_row = all_df.iloc[selected_idx]
    print("selected_row",selected_row)
    image_path = os.path.join(image_dir,selected_row.img_file+".jpg")
    selected_img = Image.open(image_path)
    st.image(selected_img, caption="You selected", width=200)

    # Get recommendations
    print("................heading for recommendation pipeline............")
    results = recommendation_pipeline(image_path, top_k=6)
    print(results)
    styles_df["id"] = styles_df["id"].astype(str)
    results["img_file"] = results["img_file"].astype(str)
    results = pd.merge(styles_df,results,left_on="id",right_on="img_file")
    print(".................got results.............")
    print(results.shape)
    
#"gender	masterCategory	subCategory	articleType	baseColour	season	year	usage	productDisplayName
#"
    # Display recommended images with tags
    print("starting to display similar images")
    rec_cols = st.columns(3)
    for j, row in enumerate(results.itertuples()):
        with rec_cols[j % 3]:
            try: 
                image_path = os.path.join(image_dir,row.img_file+".jpg")
                print(image_path)
                rec_img = Image.open(image_path)
                st.image(rec_img, use_container_width=True)
                #st.write(f"**Tags:** {row.gender}, {row.masterCategory},{row.subCategory},{row.articleType},{row.baseColour},{row.season},{row.year},{row.usage}")
                st.caption(f"Name: {row.productDisplayName} \nSimilarity: {row.curr_sim:.2f}")
            except:
                print("in exception")
                st.write(row.img_file)
