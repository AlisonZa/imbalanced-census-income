import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler
import hdbscan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA






def perform_kmeans_analysis(data: np.ndarray, k_max: int) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Perform k-means clustering analysis with different numbers of clusters
    and calculate various clustering metrics.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data for clustering (n_samples, n_features)
    k_max : int
        Maximum number of clusters to test
        
    Returns:
    --------
    Tuple containing:
        - pd.DataFrame: Metrics for each number of clusters
        - plt.Figure: Elbow plot
    """
    
    # Initialize lists to store metrics
    metrics = {
        'n_clusters': [],
        'inertia': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    # Perform k-means for different numbers of clusters
    for k in range(2, k_max + 1):
        # Fit k-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        
        # Calculate metrics
        labels = kmeans.labels_
        metrics['n_clusters'].append(k)
        metrics['inertia'].append(kmeans.inertia_)
        metrics['silhouette'].append(silhouette_score(data, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(data, labels))
    
    # Create DataFrame with metrics
    metrics_df = pd.DataFrame(metrics)
    
    # Create elbow plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot inertia (elbow curve)
    ax.plot(metrics_df['n_clusters'], metrics_df['inertia'], 
            marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Set integer x-axis ticks
    ax.set_xticks(range(2, k_max + 1))
    ax.set_xticklabels(range(2, k_max + 1))
    
    # Customize plot
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_title('Elbow Method for Optimal k', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for each point
    for k, inertia in zip(metrics_df['n_clusters'], metrics_df['inertia']):
        ax.annotate(f'k={k}', 
                   (k, inertia),
                   textcoords="offset points",
                   xytext=(0,10),
                   ha='center')
    
    plt.tight_layout()
    
    return metrics_df, fig

def perform_agglomerative_analysis(data: np.ndarray, k_max: int) -> Tuple[pd.DataFrame, plt.Figure]:
    metrics = {
        'n_clusters': [],
        'inertia': [],  # Not applicable for Agglomerative Clustering, so will be set to None
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    for k in range(2, k_max + 1):
        # Fit Agglomerative Clustering
        agglomerative = AgglomerativeClustering(n_clusters=k)
        labels = agglomerative.fit_predict(data)
        
        metrics['n_clusters'].append(k)
        metrics['inertia'].append(None)  # Inertia is not defined for this method
        metrics['silhouette'].append(silhouette_score(data, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(data, labels))
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create elbow plot (inertia is not available here, so we skip that part)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics_df['n_clusters'], metrics_df['silhouette'], 
            marker='o', linestyle='-', linewidth=2, markersize=8)
    
    ax.set_xticks(range(2, k_max + 1))
    ax.set_xticklabels(range(2, k_max + 1))
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Agglomerative Clustering', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return metrics_df, fig

def perform_spectral_analysis(data: np.ndarray, k_max: int) -> Tuple[pd.DataFrame, plt.Figure]:
    metrics = {
        'n_clusters': [],
        'inertia': [],  # Not applicable for Spectral Clustering, so will be set to None
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    for k in range(2, k_max + 1):
        # Fit Spectral Clustering
        spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
        labels = spectral.fit_predict(data)
        
        metrics['n_clusters'].append(k)
        metrics['inertia'].append(None)  # Inertia is not defined for this method
        metrics['silhouette'].append(silhouette_score(data, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(data, labels))
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create silhouette plot (inertia is not available here, so we skip that part)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics_df['n_clusters'], metrics_df['silhouette'], 
            marker='o', linestyle='-', linewidth=2, markersize=8)
    
    ax.set_xticks(range(2, k_max + 1))
    ax.set_xticklabels(range(2, k_max + 1))
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Spectral Clustering', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return metrics_df, fig

### Robust to outliers

def perform_dbscan_analysis(data: np.ndarray, eps: float, min_samples: int) -> pd.DataFrame:
    metrics = {
        'eps': [],
        'min_samples': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    # Fit DBSCAN for different epsilon values
    for eps_value in eps:
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        metrics['eps'].append(eps_value)
        metrics['min_samples'].append(min_samples)
        metrics['silhouette'].append(silhouette_score(data, labels) if len(set(labels)) > 1 else -1)  # Handle noise
        metrics['calinski_harabasz'].append(calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else -1)
        metrics['davies_bouldin'].append(davies_bouldin_score(data, labels) if len(set(labels)) > 1 else -1)
    
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df

def perform_hdbscan_analysis(
    data: np.ndarray,
    min_cluster_sizes: range,
    min_samples_range: range
) -> pd.DataFrame:
    """
    Perform HDBSCAN clustering analysis for different parameter combinations.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    min_cluster_sizes : range
        Range of minimum cluster sizes to test
    min_samples_range : range
        Range of minimum samples values to test
        
    Returns:
    --------
    Tuple[pd.DataFrame, plt.Figure]
        DataFrame with metrics for each parameter combination and visualization
    """
    metrics = {
        'min_cluster_size': [],
        'min_samples': [],
        'n_clusters': [],
        'n_noise': [],
        'silhouette': [],
        'relative_validity': []
    }
    
    # Test different parameter combinations
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_range:
            # Skip invalid combinations
            if min_samples > min_cluster_size:
                continue
                
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                gen_min_span_tree=True
            )
            
            labels = clusterer.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            
            metrics['min_cluster_size'].append(min_cluster_size)
            metrics['min_samples'].append(min_samples)
            metrics['n_clusters'].append(n_clusters)
            metrics['n_noise'].append(n_noise)
            metrics['silhouette'].append(
                silhouette_score(data, labels) if n_clusters > 1 and n_clusters < len(data) else -1
            )
            metrics['relative_validity'].append(clusterer.relative_validity_)
    
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df

def perform_robust_kmeans_analysis(data: np.ndarray, k_range: range) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Perform Robust K-means clustering analysis using RobustScaler preprocessing.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    k_range : range
        Range of k values to test
        
    Returns:
    --------
    Tuple[pd.DataFrame, plt.Figure]
        DataFrame with metrics for each k value and figure showing scores
    """
    # Apply robust scaling to handle outliers
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    
    metrics = {
        'n_clusters': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': [],
        'inertia': []
    }
    
    # Fit Robust K-means for different numbers of clusters
    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        labels = kmeans.fit_predict(scaled_data)
        
        metrics['n_clusters'].append(k)
        metrics['silhouette'].append(silhouette_score(scaled_data, labels) if k > 1 else -1)
        metrics['calinski_harabasz'].append(calinski_harabasz_score(scaled_data, labels) if k > 1 else -1)
        metrics['davies_bouldin'].append(davies_bouldin_score(scaled_data, labels) if k > 1 else -1)
        metrics['inertia'].append(kmeans.inertia_)
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette score plot
    ax1.plot(metrics_df['n_clusters'], metrics_df['silhouette'], 
             marker='o', linestyle='-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Silhouette Score', fontsize=12)
    ax1.set_title('Silhouette Score vs Number of Clusters', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Elbow plot
    ax2.plot(metrics_df['n_clusters'], metrics_df['inertia'], 
             marker='o', linestyle='-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Inertia', fontsize=12)
    ax2.set_title('Elbow Plot', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return metrics_df, fig

def plot_clustering(df, model_class, **model_params):
    """
    Plot clustering results from a dataframe and clustering model.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Pre-scaled input data
    model_class : sklearn clustering model class
        Clustering model class (e.g., KMeans, DBSCAN)
    **model_params : dict
        Parameters for the clustering model (e.g., eps, min_samples for DBSCAN)
    
    Returns:
    --------
    fig : matplotlib figure
        The generated plot
    model : fitted model instance
        The fitted clustering model
    """
    # Fit the model
    model = model_class(**model_params)
    model.fit(df)
    
    # Get labels
    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(df)
    
    # Apply PCA for visualization if needed
    if df.shape[1] > 2:
        pca = PCA(n_components=2)
        X_plot = pca.fit_transform(df)
        explained_var = pca.explained_variance_ratio_
        axis_labels = [f'PC{i+1} ({var:.1%})' for i, var in enumerate(explained_var)]
    else:
        X_plot = df.values
        axis_labels = df.columns[:2]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot points
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], 
                         c=labels, cmap='viridis', 
                         alpha=0.6)
    
    # Add colorbar
    plt.colorbar(scatter, label='Cluster')
    
    # Add labels and title
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(f'Clustering Results ({model.__class__.__name__})')
    
    # Add cluster centers if available (e.g., for KMeans)
    if hasattr(model, 'cluster_centers_'):
        centers = model.cluster_centers_
        if df.shape[1] > 2:
            centers = pca.transform(centers)
        plt.scatter(centers[:, 0], centers[:, 1], 
                   c='red', marker='x', s=200, linewidths=3,
                   label='Cluster Centers')
        plt.legend()
    
    plt.tight_layout()
    return fig, model

def analyze_clusters(raw_data: pd.DataFrame, treated_data: pd.DataFrame, clustering_algorithm) -> pd.DataFrame:
    """
    Perform descriptive analysis of clusters on the original dataset.

    Parameters:
        raw_data (pd.DataFrame): The original dataset, including missing values.
        treated_data (pd.DataFrame): The cleaned and preprocessed dataset used for clustering.
        clustering_algorithm: A clustering algorithm instance (e.g., KMeans).
    
    Returns:
        pd.DataFrame: A summary DataFrame with descriptive statistics for each cluster.
    """
    # Perform clustering on the treated data
    clustering_algorithm.fit(treated_data)
    labels = clustering_algorithm.labels_
    
    # Add cluster labels to the original data
    raw_data_with_clusters = raw_data.copy()
    raw_data_with_clusters['cluster'] = labels
    
    # Compute descriptive statistics for each cluster
    summary = raw_data_with_clusters.groupby('cluster').agg({
        column: ['mean', 'median', 'std', 'min', 'max', 'count'] for column in raw_data.columns if column != 'cluster'
    })
    
    # Flatten multi-level column index
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary






