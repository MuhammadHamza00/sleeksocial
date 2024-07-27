import streamlit as st
import os
import json
from dotenv import load_dotenv
from llamaapi import LlamaAPI
import transformers
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()

# Initialize Llama API
llama_api_token = os.getenv('LLAMA_API_KEY')
llama = LlamaAPI(llama_api_token)

# Initialize the text generation pipeline (for fallback or comparison)
# generator = pipeline("text-generation", model="gpt2")

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Function to generate posts using Llama API
def generate_posts(idea, platforms, num_posts, tags_option, language):
    posts = {platform: [] for platform in platforms}
    base_prompt = (
        f"Create {num_posts} engaging social media posts based on the following idea:\n\n"
        f"Idea: {idea}\n"
        f"Language: {language}\n"
        f"Tags: {tags_option}\n"
        f"Each post should be 300 words long.\n"
    )
    
    for platform in platforms:
        platform_prompt = base_prompt + f"\nCreate posts specifically tailored for {platform}.\n"
        api_request_json = {
            "model": "llama3-70b",
            "messages": [
                {"role": "system", "content": "You are a social media assistant."},
                {"role": "user", "content": platform_prompt},
            ]
        }
        
        for _ in range(num_posts):
            response = llama.run(api_request_json)
            generated = response.json()['choices'][0]['message']['content'].strip()
            posts[platform].append(generated)
    
    return posts

# Function to display the post creation page
def post_creation_page():
    st.title("Sleek Social")

    # Search bar in the bottom center
    st.markdown(
        """
        <style>
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="centered">', unsafe_allow_html=True)
    idea_input = st.text_input("Enter your idea here...", key="idea_input", help="Enter your idea here...")
    st.markdown('</div>', unsafe_allow_html=True)

    if idea_input:
        posts = generate_posts(idea_input, platforms, num_posts, tags_option, language)
        tabs = st.tabs(platforms)

        # Display posts for each platform
        for tab, platform in zip(tabs, platforms):
            with tab:
                st.write(f"### {platform} Posts")
                if platform in posts:
                    for i, post in enumerate(posts[platform]):
                        st.text_area(f"**Post {i+1}**: {post}")

# Function to display the sentiment analysis page
def sentiment_analysis_page():
    st.title("Sentiment Analysis")
    #  Search bar in the bottom center
    st.markdown(
        """
        <style>
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # User input for social media post
    post_input = st.text_area("Enter a social media post for sentiment analysis...", key="post_input", help="Enter a social media post here...")
    
    if post_input:
        analysis = sentiment_analyzer(post_input)[0]
        label = analysis['label']
        score = analysis['score']

        if label == 'POSITIVE':
            emoji = '😊'
        elif label == 'NEGATIVE':
            emoji = '😞'
        else:
            emoji = '😐'

        st.write(f"**Sentiment:** {label} {emoji}")
        st.write(f"**Confidence Score:** {score:.2f}")

def post_analyzer_page():
    st.title("Post Analyzer")

    # User input for the social media post
    post_input = st.text_area("Enter a social media post to analyze...", key="post_analyze_input", help="Enter a social media post here...")

    if post_input:
        # Select analysis option
        analysis_option = st.radio(
            "What would you like to do with this post?",
            ("Summarize", "Explain"),
            index=0  # default to Summarize
        )

        if st.button("Analyze"):
            # Prepare prompt for Llama API
            if analysis_option == "Summarize":
                try:
                    summary_prompt = (
                        f"Please summarize the following social media post into a concise overview, capturing the key points and main message:\n\n"
                        f"{post_input}"
                    )
                    api_request_json = {
                        "model": "llama3-70b",
                        "messages": [
                            {"role": "system", "content": "You are a social media assistant."},
                            {"role": "user", "content": summary_prompt},
                        ]
                    }
                    response = llama.run(api_request_json)
                    generated_summary = response.json()['choices'][0]['message']['content'].strip()
                    
                    st.write("### Summary:")
                    st.write(generated_summary)
                except Exception as e:
                    st.error(f"An error occurred while summarizing the post: {str(e)}")

            elif analysis_option == "Explain":
                try:
                    explanation_prompt = (
                        f"Please explain the following social media post in detail, including context, background, and any necessary insights to fully understand the content:\n\n"
                        f"{post_input}"
                    )
                    api_request_json = {
                        "model": "llama3-70b",
                        "messages": [
                            {"role": "system", "content": "You are a social media assistant."},
                            {"role": "user", "content": explanation_prompt},
                        ]
                    }
                    response = llama.run(api_request_json)
                    generated_explanation = response.json()['choices'][0]['message']['content'].strip()
                    
                    st.write("### Explanation:")
                    st.write(generated_explanation)
                except Exception as e:
                    st.error(f"An error occurred while explaining the post: {str(e)}")


# Sidebar
st.sidebar.title("Navigation")
nav_option = st.sidebar.selectbox("Select a Page", ["Post Creation", "Sentiment Analysis", "Post Analyzer"])

st.sidebar.title("Platforms Selection")
platforms = st.sidebar.multiselect(
    "Choose Platforms", ["Insta", "LinkedIn", "X", "Facebook", "Reddit"]
)

st.sidebar.title("Number of Posts")
num_posts = st.sidebar.selectbox("Select number of posts", [1, 2, 3])

st.sidebar.title("Tags")
tags_option = st.sidebar.selectbox("Include Tags?", ["Yes", "No"])

st.sidebar.title("Language")
language = st.sidebar.selectbox("Select Language", ["English", "Urdu", "Arabic"])

# Page navigation
if nav_option == "Post Creation":
    post_creation_page()
elif nav_option == "Sentiment Analysis":
    sentiment_analysis_page()
elif nav_option == "Post Analyzer":
    post_analyzer_page()
