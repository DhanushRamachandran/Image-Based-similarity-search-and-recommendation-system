# Image-Based-similarity-search-and-recommendation-system

An AI-powered Fashion Recommendation Engine that finds and recommends visually similar clothing items using ResNet feature embeddings.
This project leverages deep learning and image similarity search to enable next-generation product discovery for fashion and retail platforms.

🚀 Overview

Given a list of images searched through streamlit, an initial set of 25 images are given as output.
Once an image is selected, the system retrieves the most visually similar items from a product catalog.
It uses ResNet for deep feature extraction and FAISS (or Annoy) for efficient similarity search.
This multi-stage process replicates how users explore fashion visually — starting broad, then refining their preferences naturally.
In fashion, style is visual — customers are drawn by what they see, not just product titles or tags.
Traditional recommendation systems rely on metadata (color, brand, category), which cannot capture visual style, pattern, or texture.

👀 Fashion is visual-first — images communicate design, fit, and feel better than words.

🧭 Bridges the “intent gap” — helps users find what they want visually when they can’t describe it.

   Contextual discovery — “Show me more like this” mimics real-world browsing behavior.

🛒 Boosts engagement and conversion — users stay longer and find products that match their taste.

  Scalable personalization — works across millions of items and users without retraining per user.
This project solves that problem using deep visual embeddings extracted via ResNet, allowing the system to understand fashion images like a human stylist would.
How It Works
1) Feature Extraction

Uses ResNet50 (PyTorch) pre-trained on ImageNet.

Extracts 2048-dimensional embeddings from the final convolutional layer.

2) Indexing

Stores all product embeddings in FAISS (or Annoy) for fast similarity search.

3) Recommendation Flow

User queries or searches → top 25 visually similar items are shown.

On selecting any item → model finds other items with closest embedding distances (fine-grained visual match).

4) Streamlit Interface

Interactive gallery of items (grid view).

Click to refine → dynamically updates with similar products.

Intuitive, real-time browsing experience.

🧰 Tech Stack
Component	Technology
Language	Python 3.x
Frameworks	PyTorch, Streamlit
Feature Extractor	ResNet50 (torchvision.models)
Vector Indexing	FAISS / Annoy
Image Handling	OpenCV, Pillow
Data Management	NumPy, Pandas
Visualization	Matplotlib, Streamlit UI

# DATASET
KAGGLE: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
# CODE SNIPPETS

<img width="892" height="455" alt="image" src="https://github.com/user-attachments/assets/e7322ced-314a-4940-81ff-97a52d054327" />
# APP Run Recommendation
<img width="928" height="398" alt="image" src="https://github.com/user-attachments/assets/90cf30c2-5d94-4e0a-b92d-1e39dc9bec58" />

# CONCLUSION
This project demonstrates the power of deep learning-based feature extraction for building an effective fashion recommendation system. By leveraging ResNet as a feature extractor, we were able to convert images into high-dimensional embeddings that capture visual similarity. The system allows users to explore a catalog of fashion items by showing similar items based on visual appearance, enhancing discoverability and personalization.

# Key takeaways:

Visual similarity matters in fashion — users are more likely to engage when recommendations match the style, color, or texture of items they like.

Deep learning embeddings outperform traditional feature matching — ResNet-based features capture complex patterns that are difficult to encode manually.

The approach is scalable and modular — the feature extraction model can be used for new items without retraining, and the similarity computation can be adapted to large datasets.

Practical impact — this system can improve user experience in e-commerce, reduce decision fatigue, and increase conversion rates by providing relevant recommendations.


