import time
from sklearn.neighbors import NearestNeighbors
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
    print("Downloading images...")
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


def load_nearest_neighbors_model(features_dict, n_neighbors=10):
    # Convert the dictionary of features to a matrix
    features_matrix = np.array(list(features_dict.values()))

    # Create a NearestNeighbors object with the desired number of neighbors
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')

    # Fit the NearestNeighbors object to the feature matrix
    neighbors.fit(features_matrix)

    return neighbors



def retrieve_similar_images(features, features_dict, model):    
    # Find the indices of the closest images to the query image
    closest_indices = model.kneighbors([features], return_distance=False)[0]
    
    # Retrieve the filenames of the closest images
    closest_images = [list(features_dict.keys())[i] for i in closest_indices]
    
    return closest_images


def main():
    st.title("LFW Similarity Search App")
    # Select the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = torch.cuda.is_available()
    st.write("Computing device: ", device)
    # Loading Images
    with st.spinner("Loading Images..."):
        load_images()
    # Loading the features and the distance matrix
    with st.spinner("Loading the model, features and distance matrix..."):
        model = load_model()
        model = model.to(device)
        features_dict = load_features()
        distance_model = load_nearest_neighbors_model(features_dict)
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
        # regardless of the OS
        person_name = selected_image.split(os.path.sep)[-2]
        st.write("Person name: ", person_name)
        st.write("Image name: ", os.path.basename(selected_image))
        with st.spinner("Computing the features & retrieving similar images..."):
            features = features_dict[os.path.basename(selected_image)]
            similar_images = retrieve_similar_images(features, features_dict, distance_model)
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
            with st.spinner("Computing the features & retrieving similar images..."):
                # Process the uploaded image
                image = preprocess_image(uploaded_file)
                image = image.to(device)
                features = model(image)
                features_matrix = features.data.squeeze().cpu().numpy(
                ) if is_cuda else features.data.squeeze().numpy()
                # get the nearest neighbors
                similar_images = retrieve_similar_images(
                    features_matrix, features_dict, distance_model)
                st.write("Similar images:")
                # find the images from the folders and display them
                for image in similar_images:
                    image_path = [os.path.join(root, name) for root, dirs, files in os.walk(IMAGES_DIR+"/lfw") for name in files if name.endswith(image)]
                    st.image(image_path[0], caption=image, use_column_width=True)


if __name__ == '__main__':
    main()
