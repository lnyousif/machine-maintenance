# Import dependencies
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Run a K-means model and add its predictions as a new column to the dataframe
def run_kmeans_model(dataframe, n_clusters_choice, random_state_choice):
    # Create a copy of the dataframe
    dataframe2 = dataframe.copy()
    # Create and initialize the K-means model instance
    kmeans_model = KMeans(n_clusters = n_clusters_choice, random_state = random_state_choice)
    # Fit the data to the instance of the model
    kmeans_model.fit(dataframe2)
    # Make predictions about the data clusters using the trained model
    predictions = kmeans_model.predict(dataframe2)
    # Add a column to the DataFrame that contains the predictions information
    dataframe2["KMeans Cluster"] = predictions
    # Return dataframe
    return dataframe2


# Run a Birch model and add its predictions as a new column to the dataframe
def run_birch_model(dataframe, n_clusters_choice):
    # Create a copy of the dataframe
    dataframe2 = dataframe.copy()
    # Create and initialize the Birch model instance
    birch_model = Birch(n_clusters = n_clusters_choice)
    # Fit the data to the instance of the model
    birch_model.fit(dataframe2)
    # Make predictions about the data clusters using the trained model
    birch_predictions = birch_model.predict(dataframe2)
    # Add a column to the DataFrame that contains the predictions information
    dataframe2["Birch Cluster"] = birch_predictions
    # Return dataframe
    return dataframe2


# Run an Agglomerative model and add its predictions as a new column to the dataframe
def run_agglo_model(dataframe, n_clusters_choice):
    # Create a copy of the dataframe
    dataframe2 = dataframe.copy()
    # Create and initialize the Birch model instance
    agglo_model = AgglomerativeClustering(n_clusters = n_clusters_choice)
    # Fit the data to the instance of the model and make predictions
    agglo_predictions = agglo_model.fit_predict(dataframe2)
    # Add a column to the DataFrame that contains the predictions information
    dataframe2["Agglo Cluster"] = agglo_predictions
    # Return dataframe
    return dataframe2


# Run multiple clustering models and add their predictions as new columns to the dataframe
def run_multiple_clustering_models(dataframe, n_clusters_choice, random_state_choice):
    dataframe2 = run_kmeans_model(dataframe, n_clusters_choice, random_state_choice)
    dataframe3 = run_birch_model(dataframe2, n_clusters_choice)
    dataframe4 = run_agglo_model(dataframe3, n_clusters_choice)
    return dataframe4


# Implement the elbow method for a K-means model
def create_elbow_curve(dataframe, k_max):
    # Create an empty list to store the inertia values
    inertia = []
    # Create a list with the number of k-values to try
    k = list(range(1,k_max))
    # Create a for loop to compute the inertia with each possible value of k and add the values to the inertia list.
    for i in k:
        model = KMeans(n_clusters=i, random_state=1)
        model.fit(dataframe)
        inertia.append(model.inertia_)
    # Create a dictionary with the data to plot the elbow curve
    elbow_data = {"k": k, "inertia": inertia}
    # Create a DataFrame with the data to plot the elbow curve
    df_elbow = pd.DataFrame(elbow_data)
    # Plot the Elbow curve
    df_elbow.plot.line(x = "k", y = "inertia", title = "Elbow Curve", xticks = k)
    # Determine the rate of decrease between each k value. 
    k = elbow_data["k"]
    inertia = elbow_data["inertia"]
    for i in range(1, len(k)):
        percentage_decrease = (inertia[i-1] - inertia[i]) / inertia[i-1] * 100
        print(f"Percentage decrease from k={k[i-1]} to k={k[i]}: {percentage_decrease:.2f}%")


# Run a PCA model
def run_pca_model(dataframe, n_components_choice):
    column_names = [f'PCA{i+1}' for i in range(n_components_choice)]
    # Instantiate the PCA instance and declare the number of PCA variables
    pca = PCA(n_components = n_components_choice)
    # Fit the PCA model on the DataFrame
    pca_data = pca.fit_transform(dataframe)
    # Create the PCA DataFrame
    pca_dataframe = pd.DataFrame(pca_data, columns = column_names)
    # Determine which feature has the stronger influence on each principal component
    pca_component_weights = pd.DataFrame(pca.components_.T, columns = column_names, index = dataframe.columns)
    # Return dataframe
    return pca_dataframe, pca.explained_variance_ratio_, pca_component_weights


# Build the encode_method helper function
def encode_column(dataframe, column):
    """
    This function encodes the unique categories in the specified column of the DataFrame
    and replaces the category names with the corresponding category numbers.
    """
    # Obtain unique categories in the column
    unique_categories = dataframe[column].unique()
    # Create a category map to map category names to numbers
    category_map = {category: i + 1 for i, category in enumerate(unique_categories)}
    # Replace categories in the specified column of the DataFrame
    dataframe[f"Encoded {column}"] = dataframe[column].map(category_map)
    return category_map


# Scale the numeric columns in a dataframe
def scale_numeric_columns(dataframe):
    # Identify numeric columns
    numeric_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns
    # Scale numeric columns
    scaler = StandardScaler()
    dataframe[numeric_columns] = scaler.fit_transform(dataframe[numeric_columns])
    return dataframe


# Create dummy variables for specific column(s)
def create_dummies(dataframe, column):
    # Transform the column using get_dummies()
    dummy_columns = pd.get_dummies(dataframe[column], dtype = "int")
    # Concatenate the dummy_columns onto the original dataframe, and delete the original column that was dummied
    dataframe = pd.concat([dataframe, dummy_columns], axis = 1).drop(columns = column)
    return dataframe


# Make plots
def make_plot(dataframe, kind_value, x_value, y_value, c_value, colormap_value):
    return dataframe.plot(kind = kind_value, x = x_value, y = y_value, c = c_value, colormap = colormap_value)