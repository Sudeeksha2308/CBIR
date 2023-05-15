import os
from loguru import logger
from tqdm import tqdm
import numpy as np
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pickle 

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

class deepCBIR:
    def __init__(self):
        self.pickle_path = "./app/static/database.pkl"          # To store feature vectors of images
        self.load_cbir_model()                                  
        self.vectorize_database("./app/dataset/training_set")   # Vectorizes the images in the training data

    def load_cbir_model(self):
        # Loads the pretrained model(InceptionResNetV2) for extracting features from image      
        self.cbir_model = InceptionResNetV2(weights="imagenet", include_top=True, input_shape=(299, 299, 3))    # Includes fully connected layer
        # Sets the model's last layer as the output
        self.cbir_model = Model(inputs=self.cbir_model.input, outputs=self.cbir_model.get_layer("avg_pool").output)

    def vectorize_database(self, database_dir):
        # If there is saved pickle file, it loads features vectors from file
        try: 
            with open(self.pickle_path, "rb") as f:
                self.database = pickle.load(f)
        # If not present, extracts features from images using InceptionResNetV2 model and saves it in pickle file
        except FileNotFoundError: 
            img_paths = glob.glob(os.path.join(database_dir, '**', '*'), recursive=True)
            self.database = {}
            for img_url in tqdm(img_paths):
                print("img_url=",img_url)
                try:
                    self.database[img_url] = self.img_to_encoding(img_url, self.cbir_model)
                except:
                    pass
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.database, f)
        # Constructs feature matrix that stacks all feature vectors into rows
        if not self.database:
            print("Database is empty!")
            return

        # Constructs feature matrix that stacks all feature vectors into rows
        print("len of database value=",len(self.database.values()))
        self.features = np.array(list(self.database.values())).reshape(len(self.database), -1)


    # Resizes input image
    def img_to_encoding(self, img1, model):
        #img1 = image.load_img(image_path, target_size=(299, 299, 3))
        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        embedding = model.predict_on_batch(x)
        return embedding
    
    def retrieve_images(self, query_img_path, scope):
    # Load the query image and its four rotation variants
        query_imgs = []
        query_embeddings = []
        for angle in [0, 90, 180, 270]:
            
            img = image.load_img(query_img_path, target_size=(299, 299, 3))
            img = img.rotate(angle)
            
            x = image.img_to_array(img)
            print("shape of query_img_path:",x.shape)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            embedding = self.img_to_encoding(img, self.cbir_model)
            query_embeddings.append(embedding)
            query_imgs.append(x)

    # # Compute the embeddings for the query images and their rotation variants
    #     query_embeddings = []
    #     for img in query_imgs:
    #         embedding = self.img_to_encoding(img, self.cbir_model)
    #         query_embeddings.append(embedding)

    # Compute the cosine similarity between the query embeddings and the database embeddings
        similarities = []
        for query_embedding in query_embeddings:
            sim_vec = np.dot(self.features, query_embedding.T)
            sim_vec = sim_vec.flatten()
            sim_vec = sim_vec / np.linalg.norm(self.features, axis=1)
            sim_vec = sim_vec / np.linalg.norm(query_embedding)
            similarities.append(sim_vec)

    # Compute the maximum similarity across all four variants for each image in the database
        max_similarities = np.max(similarities, axis=0)

    # Sort the images by descending similarity and return the top-scoring ones
        idx = np.argsort(max_similarities)[::-1]
        return [list(self.database.keys())[i] for i in idx][:scope]

    
    def create_plot(self, image_paths):
        if len(image_paths) == 1:
            img = Image.open(image_paths[0])
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.set_title('Query Image')
            ax.axis("off")
            fig.savefig("./app/tmp/query.jpg")
        else:
            rows = (len(image_paths) // 5)
            if len(image_paths) % 5 != 0:
                rows += 1
            if rows == 1:
                cols = len(image_paths)
            else:
                cols = 5
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5))
            fig.suptitle('Retrieved Images', fontsize=20)
            for i in range(len(image_paths)):
                x = i % 5
                y = i // 5
                img = Image.open(image_paths[i])
                img = img.resize((299, 299))
                if rows == 1:
                    axes[x].imshow(img)
                    axes[x].axis("off")
                else:
                    axes[y, x].imshow(img)
                    axes[y, x].axis("off")

            fig.savefig("./app/tmp/retrieved.jpg")

