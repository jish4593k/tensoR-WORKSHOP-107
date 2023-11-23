import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from skimage import transform
from skimage.color import rgb2gray
import tensorflow as tf
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to load data from a file
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

# Function to perform KMeans clustering
def perform_kmeans(images):
    flat_images = [transform.resize(image, (28, 28)).flatten() for image in images]
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(flat_images)
    labels = kmeans.labels_

    # Plotly scatter plot for visualizing KMeans clustering
    fig = px.scatter(x=flat_images[:, 0], y=flat_images[:, 1], color=labels, title='KMeans Clustering')
    fig.show()

# Function to perform linear regression
def perform_linear_regression(images, labels):
    flat_images = [transform.resize(image, (28, 28)).flatten() for image in images]
    X_train, X_test, y_train, y_test = train_test_split(flat_images, labels, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Plotly scatter plot for visualizing the regression line
    fig = px.scatter(x=X_test[:, 0], y=X_test[:, 1], color=y_test)
    fig.add_trace(px.line(x=X_test[:, 0], y=model.predict(X_test), mode='lines').data[0])
    fig.update_layout(title='Linear Regression')
    fig.show()


def perform_pca(images, labels):
    flat_images = [transform.resize(image, (28, 28)).flatten() for image in images]
    pca = PCA(n_components=2)
    coordonnees = pca.fit_transform(flat_images)


      fig = px.scatter(x=coordonnees[:, 0], y=coordonnees[:, 1], color=labels, title='PCA')
    fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2')
    fig.show()


def train_neural_network(images, labels):
    flat_images = [transform.resize(image, (28, 28)).flatten() for image in images]
    X = torch.tensor(flat_images, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(28 * 28, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    messagebox.showinfo("Neural Network Training", "Neural network training complete.")


def perform_association_rule_mining():
    # Load the data using filedialog
    file_path = filedialog.askopenfilename(title="Select a file")
    if file_path:
        df = pd.read_table(file_path, sep='\t', header=0)
        records = df.astype(str).applymap(lambda x: 'nan' if x != '1' else df.columns[df.columns == x].values[0]).values.tolist()

        association_rules = apriori(records, min_support=0.05, min_confidence=0.3, min_lift=4, min_length=2)
        association_results = list(association_rules)

        for item in association_results:
            pair = item[0]
            items = [x for x in pair]

            # Convert association rule data to a DataFrame for easier visualization
            rule_df = pd.DataFrame({
                'Rule': [f"{items[0]} -> {items[1]}"],
                'Support': [item[1]],
                'Confidence': [item[2][0][2]],
                'Lift': [item[2][0][3]]
            })

            # Use Seaborn to display association rules
            sns.barplot(x='Rule', y='Support', data=rule_df)
            plt.title('Association Rule Support')
            plt.show()

            sns.barplot(x='Rule', y='Confidence', data=rule_df)
            plt.title('Association Rule Confidence')
            plt.show()

            sns.barplot(x='Rule', y='Lift', data=rule_df)
            plt.title('Association Rule Lift')
            plt.show()
root = tk.Tk()
root.title("Data Mining Toolkit")

# Load Data button
load_data_button = tk.Button(root, text="Load Data", command=lambda: load_data())
load_data_button.pack()

# KMeans Clustering button
kmeans_button = tk.Button(root, text="Perform KMeans Clustering", command=lambda: perform_kmeans(images))
kmeans_button.pack()

# Linear Regression button
linear_regression_button = tk.Button(root, text="Perform Linear Regression", command=lambda: perform_linear_regression(images, labels))
linear_regression_button.pack()

# PCA button
pca_button = tk.Button(root, text="Perform PCA", command=lambda: perform_pca(images, labels))
pca_button.pack()

# Train Neural Network button
nn_button = tk.Button(root, text="Train Neural Network", command=lambda: train_neural_network(images, labels))
nn_button.pack()

association_button = tk.Button(root, text="Perform Association Rule Mining", command=perform_association_rule_mining)
association_button.pack()


exit_button = tk.Button(root, text="Exit", command=root.destroy)
exit_button.pack()

root.mainloop()
