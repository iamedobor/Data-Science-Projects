"""
Customer Segmentation Dashboard

A comprehensive machine learning dashboard for customer segmentation
using K-Means, Hierarchical Clustering, and Gaussian Mixture Models.

Version: 1.0.0
Author: Osasere Edobor
"""

__version__ = "1.0.0"
__author__ = "Osasere Edobor"
__email__ = "your.email@example.com"

from .utils import (
    validate_data_structure,
    create_features,
    detect_outliers_iqr,
    preprocess_clustering_data,
    interpret_silhouette_score,
    interpret_davies_bouldin,
    interpret_calinski_harabasz,
    calculate_overall_cluster_quality,
    create_cluster_profile_summary,
    classify_customer_segment,
    export_segment_to_csv,
    save_waitlist_signup,
    get_color_palette
)

__all__ = [
    'validate_data_structure',
    'create_features',
    'detect_outliers_iqr',
    'preprocess_clustering_data',
    'interpret_silhouette_score',
    'interpret_davies_bouldin',
    'interpret_calinski_harabasz',
    'calculate_overall_cluster_quality',
    'create_cluster_profile_summary',
    'classify_customer_segment',
    'export_segment_to_csv',
    'save_waitlist_signup',
    'get_color_palette'
]