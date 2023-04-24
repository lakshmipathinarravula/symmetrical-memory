import time
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import pickle
import numpy as np
import urllib.request
import tarfile
import os



IMAGES_DIR = "data"

# Load the images
# # !wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
# !tar -xvf /content/lfw.tgz
def load_images():
    # Check if the images are already downloaded
    if os.path.exists(IMAGES_DIR+"/lfw"):
        print("Images already downloaded")
        return

    # download the images
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    filename = "lfw.tgz"
    urllib.request.urlretrieve(url, filename)
    tar = tarfile.open("lfw.tgz")
    tar.extractall(IMAGES_DIR)
    tar.close()


    
# Preprocess an image and compute its features

def preprocess_image(image_path):
    # Load the image and resize it to (224, 224)
    image = Image.open(image_path)
    image = image.resize((224, 224))

    # Convert the image to a tensor and normalize it
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = transformation(image).float()

    # Add a batch dimension
    image = image.unsqueeze(0)

    return image


def load_model():
    # Load the ResNet-50 model and move it to the GPU
    model = torch.hub.load("pytorch/vision:v0.10.0",
                           "resnet50", pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


def load_features():
    # Load the features dictionary
    with open("lfw_features.pkl", "rb") as f:
        features_dict = pickle.load(f)

    return features_dict


def load_distance_matrix():
    # Load the distance matrix
    # Check if the distance matrix is already downloaded
    if os.path.exists("lfw_distance_matrix.npy"):
        print("Distance matrix already downloaded")
        return np.load("lfw_distance_matrix.npy")
    # download the distance matrix
    url = "https://s3.fr-par.scw.cloud/austons-ml-bucket/public/lakshmipathi/lfw_distance_matrix.npy"
    filename = "lfw_distance_matrix.npy"
    urllib.request.urlretrieve(url, filename)
    distance_matrix = np.load("lfw_distance_matrix.npy")
    return distance_matrix



def retrieve_similar_images(query_image, features_dict, distance_matrix):
    # Retrieve the feature vector of the query image
    query_features = features_dict[query_image]
    
    # Compute the distances between the query image and all other images
    distances = distance_matrix[list(features_dict.keys()).index(query_image)]
            
    # Sort the distances in ascending order and retrieve the indices of the 10 closest images
    closest_indices = np.argsort(distances)[:10]
    
    # Retrieve the filenames of the closest images
    closest_images = [list(features_dict.keys())[i] for i in closest_indices]
    
    return closest_images


def main():
    st.title("LFW Similarity Search App")
    # Select the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = torch.cuda.is_available()
    # Loading Images
    with st.spinner("Loading Images..."):
        load_images()
    # Loading the features and the distance matrix
    with st.spinner("Loading the model, features and distance matrix..."):
        model = load_model()
        model = model.to(device)
        features_dict = load_features()
        distance_matrix = load_distance_matrix()
        # time.sleep(5)

    # Define the sidebar with the two options: select an image from the test dataset or upload an image
    option = st.sidebar.selectbox(
        "Select an option", ["Select from test dataset", "Upload an image"])
    if option == "Select from test dataset":
        st.write("Selected option: select from test dataset")
        # Define the dropdown menu with the list of test images
        
        test_images = [os.path.join(root, name) for root, dirs, files in os.walk(IMAGES_DIR+"/lfw") for name in files if name.endswith(".jpg")]
        selected_image = st.selectbox("Select an image", test_images)
        # Display the selected image and retrieve the 10 most similar images
        st.image(selected_image, caption="Selected image", use_column_width=True)
        # get the person name from the image path, e.g. "data1/lfw\Aaron_Eckhart\Aaron_Eckhart_0001.jpg" -> "Aaron_Eckhart"
        person_name = selected_image.split("\\")[-2]
        st.write("Person name: ", person_name)
        st.write("Image name: ", os.path.basename(selected_image))
        similar_images = retrieve_similar_images(os.path.basename(selected_image), features_dict, distance_matrix)
        st.write("Similar images:")
        # find the images from the folders and display them
        for image in similar_images:
            image_path = [os.path.join(root, name) for root, dirs, files in os.walk(IMAGES_DIR+"/lfw") for name in files if name.endswith(image)]
            st.image(image_path[0], caption=image, use_column_width=True)

    else:
        # Define the file uploader to upload an image
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png"])

        st.write("Selected option: upload an image")

        if uploaded_file is not None:
            # Preprocess the uploaded image and extract its feature vector
            image = Image.open(uploaded_file)
            # Show the uploaded image
            st.image(image, caption="Uploaded image", use_column_width=True)
            # Process the uploaded image
            image = preprocess_image(uploaded_file)
            image = image.to(device)
            features = model(image)
            features_matrix = features.data.squeeze().cpu().numpy(
            ) if is_cuda else features.data.squeeze().numpy()
            # Compute the distances between the uploaded image and top 10 similar images
            # distances = np.linalg.norm(features_matrix - distance_matrix, axis=1)
            # closest_indices = np.argsort(distances)[:10]
            # closest_images = [list(features_dict.keys())[i] for i in closest_indices]
            # # Display the uploaded image and the 10 most similar images
            # st.image(image, caption="Uploaded image", use_column_width=True)
            # st.write("Similar images:")
            # print(closest_images)

        # Compute the distances between the uploaded image and all other


if __name__ == '__main__':
    main()
