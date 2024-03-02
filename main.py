import numpy as np
import pandas as pd
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier

# Function to load the dataset from CSV file
def load_dataset(csv_file):
    data = pd.read_csv(csv_file)
    image_paths = data['Filename'].tolist()
    labels = data['Label'].tolist()
    return image_paths, labels

# Function to preprocess images
def preprocess_images(image_paths):
    preprocessed_images = []
    for path in image_paths:
        image = cv2.imread(path)
        # Resize the image to a fixed size (e.g., 100x100 pixels)
        image = cv2.resize(image, (100, 100))
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Flatten the image array to 1D
        image = image.flatten()
        # Normalize the pixel values to the range [0, 1]
        image = image / 255.0
        preprocessed_images.append(image)
    return preprocessed_images

# Function to train the KNN model
def train_knn_model(images, labels):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(images, labels)
    return knn_model

# Function to generate new images
def generate_images(knn_model, conditions):
    generated_images = []
    for condition in conditions:
        # Assuming the conditions are features used for prediction
        generated_image = knn_model.predict([condition])
        generated_images.append(generated_image)
    return generated_images

# Main function
def main():
    # Load the dataset
    csv_file = 'annotations.csv'
    image_paths, labels = load_dataset(csv_file)
    
    # Preprocess images
    images = preprocess_images(image_paths)
    
    # Train the KNN model
    knn_model = train_knn_model(images, labels)
    
    # Generate new images based on conditions
    conditions = [[0.5, 0.3, 0.2]]  # Example condition, replace with your own
    generated_images = generate_images(knn_model, conditions)
    
    # Display or save the generated images
    for image in generated_images:
        print("Generated Image:", image)  # Example, replace with your own display or save code

if __name__ == "__main__":
    main()
