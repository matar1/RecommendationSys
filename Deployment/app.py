#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gradio as gr

import nltk
from nltk.corpus import stopwords
 
#nltk.download('stopwords')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


#Read data
all_dishes = pd.read_csv('all_dishes.csv')


tfid = TfidfVectorizer()
tfid_matrix = tfid.fit_transform(all_dishes['describe'])


# In[20]:


#cosine similarity 

cosine_sim = cosine_similarity(tfid_matrix, tfid_matrix)


# In[21]:


#(683, 683) meaning we have computed a similarity score for each pair of dishes in our dataset.


# In[22]:


def rank(title, cosine_sim = cosine_sim, dish_matrix = all_dishes):
    title = title.lower()
    idx = dish_matrix.index[dish_matrix['name'] == title].tolist()
    match_food = np.array(cosine_sim[idx[0]])
    indices = np.argsort(match_food)[-6:][::-1][1:]                #get the index of the top 5 similar dishes in decendening order
    
    return indices
    
def print_similar_dishes(title, indices = None, dish_matrix = all_dishes):
    indices = rank(title, cosine_sim = cosine_sim, dish_matrix = all_dishes)
    print("Top 5 similar dishes to", title, "are:")
    for i in range(0,len(indices)):
        print(i+1, ". ", dish_matrix['name'].iloc[indices[i]])
        
def rank_for_gradio(title):
    indices = rank(title)
    similar_dishes = [all_dishes.iloc[i]['name'] for i in indices]
    formatted_dishes = "\n".join([f"{i+1}. {dish}" for i, dish in enumerate(similar_dishes)])
    return formatted_dishes
   





# ## Front-end

# In[29]:


# Setup Gradio interface
#css_code = ".gradio-container {background: url(https://cdn.pixabay.com/photo/2024/06/22/08/07/ai-generated-8845695_1280.jpg)}"
# css_code = ".gradio-container {background: url(https://img.freepik.com/free-photo/elevated-view-ingredients-dryfruits-vegetables-black-background_23-2148026898.jpg?w=1480&t=st=1727902059~exp=1727902659~hmac=4f7f4e8f697f78d214ab8f9ee921baa16a01f17683c9b1f44779effcdb546ef9)}"

# iface = gr.Interface(
#     fn=rank_for_gradio,
#     inputs=gr.Dropdown(list(all_dishes["name"]), label="Choose a Dish"),
#     outputs=gr.Text(label="Top 5 Similar Dishes that you might like"),
#     title="Personalized Dish Recommender",
#     description="Select a dish to find the top 5 similar dishes based on their descriptions.",
#     css=css_code,
# )
css_code = """
.gradio-container {
    background: url(https://cdn.pixabay.com/photo/2024/06/22/08/07/ai-generated-8845695_1280.jpg);
}

/* Styling for the title and description box */
.title-desc-box {
    background-color: #f8f9fa;  /* Light grey background */
    border: 1px solid #dee2e6;  /* Grey border */
    border-radius: 10px;        /* Rounded corners */
    padding: 20px;              /* Padding around text */
    margin: 20px;               /* Margin around the box */
}

/* Custom title styling */
h1, h2 {
    color: #333;                /* Dark grey text color */
    text-align: center;         /* Center-align text */
}

/* Custom description styling */
p {
    color: #555;                /* Medium grey text color */
    font-size: 16px;            /* Larger font size */
}
"""

# Interface initialization with modified css
iface = gr.Interface(
    fn=rank_for_gradio,
    inputs=gr.Dropdown(list(all_dishes["name"]), label="Choose a Dish"),
    outputs=gr.Text(label="Top 5 Similar Dishes that you might like"),
    title="<div class='title-desc-box'><h1>Personalized Dish Recommender</h1><p>Select a dish to find the top 5 similar dishes based on their descriptions.</p></div>",
    description="",
    css=css_code,
)


# In[30]:


# Launch the interface
iface.launch(share=True)


# ### Gradio Deployme