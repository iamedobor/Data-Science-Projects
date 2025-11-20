"""
Utility functions for Customer Segmentation Dashboard
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Optional


def validate_data_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate if uploaded data has the required structure for clustering
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check minimum rows
    if len(df) < 10:
        issues.append("Dataset has fewer than 10 rows")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        issues.append("Dataset needs at least 2 numeric columns for clustering")
    
    # Check for excessive missing values
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 50:
        issues.append(f"Dataset has {missing_pct:.1f}% missing values (>50%)")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for customer segmentation
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with additional features
    """
    df_processed = df.copy()
    
    # Create Age from Year_Birth if exists
    if 'Year_Birth' in df.columns and 'Age' not in df.columns:
        current_year = 2025
        df_processed['Age'] = current_year - df_processed['Year_Birth']
    
    # Create Total_Spending from Mnt* columns
    spending_cols = [col for col in df.columns if col.startswith('Mnt')]
    if len(spending_cols) > 0 and 'Total_Spending' not in df.columns:
        df_processed['Total_Spending'] = df_processed[spending_cols].sum(axis=1)
    
    # Create Total_Purchases from purchase columns
    purchase_cols = [col for col in df.columns if 'Purchase' in col and 'Num' in col]
    if len(purchase_cols) > 0 and 'Total_Purchases' not in df.columns:
        df_processed['Total_Purchases'] = df_processed[purchase_cols].sum(axis=1)
    
    # Create Total_Children from Kidhome and Teenhome
    if 'Kidhome' in df.columns and 'Teenhome' in df.columns and 'Total_Children' not in df.columns:
        df_processed['Total_Children'] = df_processed['Kidhome'] + df_processed['Teenhome']
    
    return df_processed


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        df: Input dataframe
        column: Column name to check for outliers
        
    Returns:
        Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def preprocess_clustering_data(
    df: pd.DataFrame,
    selected_features: List[str],
    handle_missing: str = "Fill with Median",
    remove_duplicates: bool = False,
    remove_outliers: bool = False,
    outlier_features: Optional[List[str]] = None
) -> Tuple[np.ndarray, pd.DataFrame, pd.Index]:
    """
    Preprocess data for clustering
    
    Args:
        df: Input dataframe
        selected_features: List of features to use for clustering
        handle_missing: Method to handle missing values
        remove_duplicates: Whether to remove duplicate rows
        remove_outliers: Whether to remove outliers
        outlier_features: Features to check for outliers
        
    Returns:
        Tuple of (scaled_data, processed_df, original_indices)
    """
    df_cluster = df[selected_features].copy()
    original_indices = df_cluster.index
    
    # Remove duplicates
    if remove_duplicates:
        df_cluster = df_cluster.drop_duplicates()
        original_indices = df_cluster.index
    
    # Remove constant features
    for col in df_cluster.columns:
        if df_cluster[col].std() == 0 or df_cluster[col].nunique() == 1:
            df_cluster = df_cluster.drop(columns=[col])
    
    # Handle missing values
    if handle_missing == "Fill with Median":
        df_cluster = df_cluster.fillna(df_cluster.median())
    elif handle_missing == "Fill with Mean":
        df_cluster = df_cluster.fillna(df_cluster.mean())
    else:  # Drop Rows
        df_cluster = df_cluster.dropna()
        original_indices = df_cluster.index
    
    # Remove outliers
    if remove_outliers and outlier_features:
        outlier_mask = pd.Series(True, index=df_cluster.index)
        
        for col in outlier_features:
            if col in df_cluster.columns:
                Q1 = df_cluster[col].quantile(0.25)
                Q3 = df_cluster[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_mask &= (df_cluster[col] >= lower) & (df_cluster[col] <= upper)
        
        df_cluster = df_cluster[outlier_mask]
        original_indices = df_cluster.index
    
    # Scale features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    
    return df_scaled, df_cluster, original_indices


def interpret_silhouette_score(score: float) -> Tuple[str, str, str]:
    """
    Interpret silhouette score
    
    Args:
        score: Silhouette score (-1 to 1)
        
    Returns:
        Tuple of (assessment, color, description)
    """
    if score > 0.7:
        return "Excellent", "green", "Clusters are well-separated and distinct"
    elif score > 0.5:
        return "Good", "green", "Clusters are reasonably separated"
    elif score > 0.3:
        return "Fair", "orange", "Clusters overlap somewhat but are distinguishable"
    else:
        return "Poor", "red", "Clusters overlap significantly"


def interpret_davies_bouldin(score: float) -> Tuple[str, str, str]:
    """
    Interpret Davies-Bouldin index
    
    Args:
        score: Davies-Bouldin score (0 to ∞, lower is better)
        
    Returns:
        Tuple of (assessment, color, description)
    """
    if score < 1.0:
        return "Excellent", "green", "Very distinct clusters"
    elif score < 2.0:
        return "Good", "green", "Well-separated clusters"
    elif score < 3.0:
        return "Fair", "orange", "Moderate separation"
    else:
        return "Poor", "red", "Clusters not well-separated"


def interpret_calinski_harabasz(score: float) -> Tuple[str, str, str]:
    """
    Interpret Calinski-Harabasz score
    
    Args:
        score: Calinski-Harabasz score (0 to ∞, higher is better)
        
    Returns:
        Tuple of (assessment, color, description)
    """
    if score > 1000:
        return "Excellent", "green", "Strong cluster structure"
    elif score > 500:
        return "Good", "green", "Clear cluster structure"
    elif score > 200:
        return "Fair", "orange", "Moderate cluster structure"
    else:
        return "Poor", "red", "Weak cluster structure"


def calculate_overall_cluster_quality(
    silhouette: float,
    davies_bouldin: float,
    calinski: float
) -> Tuple[float, str, str]:
    """
    Calculate overall cluster quality score
    
    Args:
        silhouette: Silhouette score
        davies_bouldin: Davies-Bouldin index
        calinski: Calinski-Harabasz score
        
    Returns:
        Tuple of (overall_score, assessment, recommendation)
    """
    # Normalize metrics to 0-1 scale
    sil_norm = max(0, min(1, (silhouette + 1) / 2))
    db_norm = max(0, min(1, 1 - (davies_bouldin / 5)))
    ch_norm = max(0, min(1, calinski / 2000))
    
    overall_score = (sil_norm + db_norm + ch_norm) / 3
    
    if overall_score > 0.6:
        assessment = "Good"
        recommendation = "Your clusters are meaningful and suitable for customer segmentation."
    elif overall_score > 0.4:
        assessment = "Fair"
        recommendation = "Consider trying different features or number of clusters."
    else:
        assessment = "Poor"
        recommendation = "Try different preprocessing, features, or algorithms."
    
    return overall_score, assessment, recommendation


def create_cluster_profile_summary(
    df: pd.DataFrame,
    cluster_col: str,
    features: List[str]
) -> pd.DataFrame:
    """
    Create a summary profile for each cluster
    
    Args:
        df: Dataframe with cluster assignments
        cluster_col: Name of cluster column
        features: Features to summarize
        
    Returns:
        DataFrame with cluster profiles
    """
    profile = df.groupby(cluster_col)[features].agg(['mean', 'median', 'std', 'count'])
    return profile


def classify_customer_segment(
    avg_spending: float,
    spending_quantiles: Dict[str, float]
) -> Tuple[str, str, str]:
    """
    Classify customer segment based on spending
    
    Args:
        avg_spending: Average spending for the cluster
        spending_quantiles: Dictionary with quantile values
        
    Returns:
        Tuple of (segment_name, emoji, description)
    """
    if avg_spending > spending_quantiles['q75']:
        return "Premium Customers", "🌟", "High-value customers with significant spending power"
    elif avg_spending > spending_quantiles['q33']:
        return "Standard Customers", "💼", "Core customer base with moderate spending"
    else:
        return "Budget Shoppers", "🛒", "Price-sensitive customers with limited spending"


def export_segment_to_csv(df: pd.DataFrame, segment_name: str) -> bytes:
    """
    Export a customer segment to CSV format
    
    Args:
        df: Dataframe containing segment data
        segment_name: Name of the segment
        
    Returns:
        CSV data as bytes
    """
    return df.to_csv(index=False).encode('utf-8')


def save_waitlist_signup(
    name: str,
    email: str,
    company: str,
    use_case: str,
    filepath: str = "../results/v2_waitlist.csv"
) -> bool:
    """
    Save v2 waitlist signup to CSV
    
    Args:
        name: User's name
        email: User's email
        company: User's company
        use_case: Primary use case
        filepath: Path to save waitlist CSV
        
    Returns:
        True if successful, False otherwise
    """
    import csv
    from datetime import datetime
    
    try:
        # Check if file exists
        file_exists = False
        try:
            with open(filepath, 'r') as f:
                file_exists = True
        except FileNotFoundError:
            pass
        
        # Append to file
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Name', 'Email', 'Company', 'Use Case'])
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                name, email, company, use_case
            ])
        
        return True
    except Exception as e:
        print(f"Error saving waitlist: {e}")
        return False


# Color palettes for consistent styling
COLOR_PALETTES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'viridis': px.colors.sequential.Viridis,
    'plasma': px.colors.sequential.Plasma,
    'coolwarm': ['#3b4cc0', '#7396f5', '#b4d7f5', '#f5c3a9', '#eb6c5c', '#b40426']
}


def get_color_palette(name: str = 'default') -> List[str]:
    """
    Get color palette by name
    
    Args:
        name: Palette name
        
    Returns:
        List of color codes
    """
    return COLOR_PALETTES.get(name, COLOR_PALETTES['default'])