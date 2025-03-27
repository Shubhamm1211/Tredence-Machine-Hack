# Databricks notebook source
!pip install datasets transformers torch accelerate -q

# COMMAND ----------

pip install protobuf==4.25.3


# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

pip install datasets transformers torch accelerate


# COMMAND ----------

from datasets import load_dataset

# Load a specific subset of the dataset
dataset = load_dataset("osunlp/TravelPlanner", "train")  # Change to "validation" or "test" as needed

# Check the dataset structure
print(dataset)



# COMMAND ----------

print(dataset["train"][0])


# COMMAND ----------

!pip install --upgrade fsspec
!pip install datasets transformers torch accelerate -q


# COMMAND ----------

# Display basic information
print(dataset)

# Show a few sample rows from the dataset
print(dataset["train"][0])  # Change "train" to "validation" or "test" if needed

# Display column names
print(dataset["train"].column_names)

# Convert to Pandas DataFrame for better visualization
import pandas as pd

df = pd.DataFrame(dataset["train"])
print(df.head())  # Show first 5 rows




# COMMAND ----------

# Example: Remove unwanted columns (if needed)
dataset = dataset.remove_columns(["date"])  # Example column removal

# Example: Convert text to lowercase (if needed)
def preprocess(example):
    example["query"] = example["query"].lower()
    return example

dataset = dataset.map(preprocess)


# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Phi-3.5 Mini tokenizer
model_checkpoint = "microsoft/Phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load Phi-3.5 Mini model
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

# Example tokenization
sample_text = "Please plan a trip from New York to Paris for 5 days."
tokens = tokenizer(sample_text, return_tensors="pt")

print(tokens.input_ids)


# COMMAND ----------

# Install necessary libraries (Run this separately in Databricks)
# %pip install transformers torch 

# Import required modules
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Phi-2 model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a travel itinerary
def generate_itinerary(destination, days, interests, budget):
    prompt = f"Generate a {days}-day travel itinerary for {destination}. The itinerary should include {interests}. Budget preference: {budget}."

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate text
    output = model.generate(input_ids, max_length=500, temperature=0.7, top_p=0.9)

    # Decode output
    itinerary = tokenizer.decode(output[0], skip_special_tokens=True)

    return itinerary

# User inputs (Use Databricks widgets for interactive input)
dbutils.widgets.text("destination", "Paris")
dbutils.widgets.text("days", "3")
dbutils.widgets.text("interests", "sightseeing, food, adventure")
dbutils.widgets.text("budget", "mid-range")

# Get values from Databricks widgets
destination = dbutils.widgets.get("destination")
days = dbutils.widgets.get("days")
interests = dbutils.widgets.get("interests")
budget = dbutils.widgets.get("budget")

# Generate itinerary
print("\n🔹 Generating your travel itinerary...\n")
itinerary = generate_itinerary(destination, days, interests, budget)
print(itinerary)

# Save itinerary to a text file in Databricks workspace
file_name = "/dbfs/FileStore/travel_itinerary.txt"
with open(file_name, "w") as file:
    file.write(itinerary)

# Provide download link for Databricks
print(f"\n✅ Itinerary saved! Download here: /files/travel_itinerary.txt")


# COMMAND ----------

# Import required modules
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Phi-2 model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a travel itinerary
def generate_itinerary(destination, days, interests, budget):
    prompt = f"Generate a {days}-day travel itinerary for {destination}. The itinerary should include {interests}. Budget preference: {budget}."

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate text
    output = model.generate(input_ids, max_length=500, temperature=0.7, top_p=0.9)

    # Decode output
    itinerary = tokenizer.decode(output[0], skip_special_tokens=True)

    return itinerary

# Databricks widgets for interactive input
dbutils.widgets.text("destination", "Paris")
dbutils.widgets.text("days", "3")
dbutils.widgets.text("interests", "sightseeing, food, adventure")
dbutils.widgets.text("budget", "mid-range")

# Retrieve values from widgets
destination = dbutils.widgets.get("destination")
days = dbutils.widgets.get("days")
interests = dbutils.widgets.get("interests")
budget = dbutils.widgets.get("budget")

# Generate itinerary
print("\n🔹 Generating your travel itinerary...\n")
itinerary = generate_itinerary(destination, days, interests, budget)
print(itinerary)

# Save itinerary to DBFS (Databricks File System)
file_path = "/dbfs/FileStore/travel_itinerary.txt"
with open(file_path, "w") as file:
    file.write(itinerary)

# Display download link
print(f"\n✅ Itinerary saved! Download here: https://<your-databricks-instance>/files/travel_itinerary.txt")


# COMMAND ----------

# Import required modules
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Phi-2 model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a travel itinerary
def generate_itinerary(destination, days, interests, budget):
    prompt = f"Generate a {days}-day travel itinerary for {destination}. The itinerary should include {interests}. Budget preference: {budget}."

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate text
    output = model.generate(input_ids, max_length=500, temperature=0.7, top_p=0.9)

    # Decode output
    itinerary = tokenizer.decode(output[0], skip_special_tokens=True)

    return itinerary

# Databricks widgets for interactive input
dbutils.widgets.text("destination", "Paris")
dbutils.widgets.text("days", "3")
dbutils.widgets.text("interests", "sightseeing, food, adventure")
dbutils.widgets.text("budget", "mid-range")

# Retrieve values from widgets
destination = dbutils.widgets.get("destination")
days = dbutils.widgets.get("days")
interests = dbutils.widgets.get("interests")
budget = dbutils.widgets.get("budget")

# Generate itinerary
print("\n🔹 Generating your travel itinerary...\n")
itinerary = generate_itinerary(destination, days, interests, budget)
print(itinerary)

# Save itinerary to DBFS (Databricks File System)
file_path = "/dbfs/FileStore/travel_itinerary.txt"
with open(file_path, "w") as file:
    file.write(itinerary)

# Display download link
print(f"\n✅ Itinerary saved! Download here: https://<your-databricks-instance>/files/travel_itinerary.txt")


# COMMAND ----------

