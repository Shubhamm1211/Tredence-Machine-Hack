import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from time import time

# Load the Phi-2 model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_itinerary(destination, days, interests, budget):
    prompt = f"Generate a {days}-day travel itinerary for {destination}. The itinerary should include {interests}. Budget preference: {budget}."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    start_time = time()
    
    # Generate the output, setting a reasonable max_length and timeout
    output = model.generate(input_ids, max_length=300, temperature=0.7, top_p=0.9)
    
    # Measure time taken to generate response
    time_taken = time() - start_time
    print(f"Time taken: {time_taken} seconds")
    
    itinerary = tokenizer.decode(output[0], skip_special_tokens=True)
    return itinerary

# Streamlit UI
st.title("🧳 Travel Itinerary Generator")

# User Inputs
destination = st.text_input("Destination", "Paris")
days = st.number_input("Number of Days", min_value=1, max_value=30, value=3)
interests = st.text_area("Interests (comma-separated)", "sightseeing, food, adventure")
budget = st.selectbox("Budget Preference", ["low", "mid-range", "luxury"], index=1)

if st.button("Generate Itinerary"):
    with st.spinner("🔹 Generating your travel itinerary..."):
        itinerary = generate_itinerary(destination, days, interests, budget)
        st.text_area("Generated Itinerary", itinerary, height=300)

        # Save itinerary as a text file
        file_path = "travel_itinerary.txt"
        with open(file_path, "w") as file:
            file.write(itinerary)
        
        # Provide download link
        with open(file_path, "rb") as file:
            st.download_button(label="Download Itinerary", data=file, file_name="travel_itinerary.txt", mime="text/plain")
