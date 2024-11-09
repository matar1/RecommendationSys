# Imports

import pandas as pd
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load and preprocess data
def preprocess(all_dishes):
    tfid = TfidfVectorizer()
    tfid_matrix = tfid.fit_transform(all_dishes['describe'])
    cosine_sim = cosine_similarity(tfid_matrix, tfid_matrix)
    return all_dishes, cosine_sim

# Function to rank similar dishes
def rank(title, cosine_sim, dish_matrix):
    title = title.lower()
    idx = dish_matrix.index[dish_matrix['name'].str.lower() == title].tolist()
    if not idx:
        return []
    match_food = np.array(cosine_sim[idx[0]])
    indices = np.argsort(match_food)[-6:][::-1][1:]  # Get the top 5 similar dishes
    return indices

# Function to format ranked dishes for display
def rank_for_gradio(title, all_dishes):
    all_dishes, cosine_sim = preprocess(all_dishes)
    indices = rank(title, cosine_sim, all_dishes)
    similar_dishes = [all_dishes.iloc[i]['name'] for i in indices]
    formatted_dishes = "\n".join([f"{i+1}. {dish}" for i, dish in enumerate(similar_dishes)])
    return formatted_dishes

# Function to dynamically update dropdown options based on halal flag
def update_dropdown(halal_flag):
    if halal_flag:
        all_dishes = pd.read_csv('all_dishes_halal.csv')
    else:
        all_dishes = pd.read_csv('all_dishes.csv')
    return gr.Dropdown(choices=list(all_dishes["name"])), all_dishes

# /* Custom CSS for interface styling */
css_code = """
.gradio-container {
    background: url(https://i.postimg.cc/FFM9WDS0/1d1ac3c2-8063-451e-bde8-0282d614c3df.webp);
    background-size: cover; /* Ensures the image covers the container */
}
/* Styling for the title and description box */
.title-desc-box {
    background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent white background */
    border: 1px solid #dee2e6;  /* Grey border */
    border-radius: 10px;        /* Rounded corners */
    padding: 20px;              /* Padding around text */
    margin: 20px;               /* Margin around the box */
}
/* Custom title styling */
.gradio-container h1, h2 {
    background: rgba(0, 0, 0, 0.5); /* Semi-transparent black background */
    text-align: center;         /* Center-align text */
    padding: 10px;
    border-radius: 5px;
    display: inline-block;
}
/* Custom description styling */
p {
    color: #555;                /* Medium grey text color */
    font-size: 16px;            /* Larger font size */
}
"""


# Interface setup with updated inputs and interactive logic
with gr.Blocks(css=css_code) as iface:
    gr.Markdown("<div class='title-desc-box'><h1>Personalized Dish Recommender</h1><br><h2>Select a dish to find the top 5 similar dishes based on their descriptions.</h2></div>")
    
    # Initial load
    all_dishes = pd.read_csv('all_dishes.csv')
    
    halal_checkbox = gr.Checkbox(label="Halal Options")
    dish_dropdown = gr.Dropdown(choices=list(all_dishes["name"]), label="Choose a Dish")
    
    # Update dropdown choices based on halal flag
    halal_checkbox.change(fn=update_dropdown, inputs=halal_checkbox, outputs=[dish_dropdown, gr.State()])

    # Button to get recommendations
    recommend_button = gr.Button("Get Recommendations")
    output_text = gr.Textbox(label="Top 5 Similar Dishes that you might like")
    recommend_button.click(fn=lambda title, halal: rank_for_gradio(title, pd.read_csv('all_dishes_halal.csv') if halal else pd.read_csv('all_dishes.csv')), inputs=[dish_dropdown, halal_checkbox], outputs=output_text)

# Launch the interface
iface.launch(share=True)
