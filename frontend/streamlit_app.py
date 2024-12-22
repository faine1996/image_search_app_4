import streamlit as st
import requests
from PIL import Image
import os

IMAGES_DIR = './images/'

# Streamlit App Title
st.title("Image Search App")

# Query Input Section
st.header("Search for Similar Images")
query = st.text_input("Enter your search query:")

# Ensure k is at least 5
k = st.number_input(
    "Number of results to return (minimum 5):",
    min_value=5,  # Minimum value is 5
    max_value=100,  # Adjust max value as needed
    value=5,  # Default value is 5
    step=1
)

if st.button("Search"):
    if query:
        search_url = "http://vector_db:5001/search"
        try:
            # Include 'k' parameter in the API request
            response = requests.post(search_url, json={"query_text": query, "k": k})
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    st.write("Search Results:")
                    
                    # Display images in a grid
                    cols = st.columns(3)  # Adjust the number of columns as needed
                    for idx, result in enumerate(results):
                        image_name = result.get("image_name", "Unknown")
                        similarity = result.get("similarity", "N/A")

                        if image_name:
                            image_path = os.path.join(IMAGES_DIR, image_name)  # Update with your image directory path
                            if os.path.exists(image_path):
                                try:
                                    # Load and display the image in a column
                                    with cols[idx % 3]:  # Distribute images across columns
                                        image = Image.open(image_path)
                                        st.image(image, caption=f"Similarity: {similarity:.4f}")
                                except Exception as e:
                                    print(image_path)
                                    st.error(f"Error loading image {image_name}: {e}")
                            else:
                                st.warning(f"Image file not found: {image_name}")
                        else:
                            st.warning("No image name provided in result.")
                else:
                    st.info("No results found.")
            else:
                st.error(f"Search failed: {response.text}")
        except Exception as e:
            st.error(f"Error contacting API: {e}")
    else:
        st.warning("Please enter a query to search.")
