# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard by Osasere Edobor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'clusters_computed' not in st.session_state:
    st.session_state.clusters_computed = False


# Sidebar
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.markdown("---")

    
    menu = st.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload Data", "📊 EDA", "🎯 Clustering", "📥 Download Results", "📖 Glossary"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This dashboard performs customer segmentation using K-Means, Hierarchical, and GMM clustering algorithms.")

# HOME PAGE
if menu == "🏠 Home":
    # Version Banner
    st.info("📌 **Version 1.0** - Specialized for Marketing & E-commerce Customer Data")
    
    st.markdown('<h1 class="main-header">📊 Customer Segmentation Dashboard by Osasere Edobor</h1>', unsafe_allow_html=True)
    
    # Main description
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Purpose")
        st.write("""
        Segment customers into distinct groups based on their:
        - Demographics (Age, Income)
        - Spending behavior
        - Purchase patterns
        - Engagement metrics
        """)
    
    with col2:
        st.markdown("### 🔧 Features")
        st.write("""
        - Upload customer CSV data
        - Exploratory Data Analysis
        - 3 clustering algorithms
        - Interactive visualizations
        - Download segmented results
        """)
    
    with col3:
        st.markdown("### 📈 Benefits")
        st.write("""
        - Targeted marketing campaigns
        - Personalized customer experiences
        - Resource optimization
        - Improved ROI
        """)
    
    st.markdown("---")
    
    # Data Requirements Section
    st.markdown("## 📋 Data Requirements for Version 1.0")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This version is optimized for **marketing and e-commerce datasets** with the following structure:
        
        **Required/Expected Columns:**
        - `Year_Birth` or `Age` - Customer age information
        - `Income` - Annual income
        - `Mnt*` columns (MntWines, MntMeatProducts, etc.) - Spending by category
        - `Num*Purchase` columns - Purchase counts by channel
        - `Recency` - Days since last purchase
        - `NumWebVisitsMonth` - Website engagement
        
        **Works Best With:**
        - 🛒 E-commerce platforms
        - 🏪 Retail customer databases
        - 📧 Email marketing lists
        - 💳 Subscription services
        """)
        
        st.warning("⚠️ **Note:** Version 1.0 requires data in this specific format. For flexible column mapping and custom datasets, see Version 2.0 below.")
    
    with col2:
        st.markdown("### 📥 Sample Dataset")
        st.info("Don't have compatible data? Download our sample dataset to test the dashboard!")
        
        try:
            with open("../data/customer_segmentation.csv", "rb") as f:
                st.download_button(
                    label="⬇️ Download Sample Data",
                    data=f.read(),
                    file_name="sample_customer_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except FileNotFoundError:
            st.info("Sample data file not found in expected location.")
    
    st.markdown("---")
    
    # Quick Start Guide
    st.markdown("### 🚀 Quick Start Guide")
    
    steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
    
    with steps_col1:
        st.markdown("""
        **Step 1️⃣**
        
        📤 Upload Data
        
        Go to 'Upload Data' and upload your customer CSV file
        """)
    
    with steps_col2:
        st.markdown("""
        **Step 2️⃣**
        
        📊 Explore
        
        View EDA to understand your data and check quality
        """)
    
    with steps_col3:
        st.markdown("""
        **Step 3️⃣**
        
        🎯 Cluster
        
        Run algorithms to segment customers into groups
        """)
    
    with steps_col4:
        st.markdown("""
        **Step 4️⃣**
        
        📥 Download
        
        Export results for marketing campaigns
        """)
    
    st.markdown("---")
    
    # Version 2.0 Teaser
    st.markdown("## 🚀 Coming Soon: Version 2.0")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 10px; 
                color: white;
                margin: 1rem 0;'>
        <h2 style='color: white; margin-top: 0;'>✨ Next Generation Customer Segmentation</h2>
        <p style='font-size: 1.1rem; margin-bottom: 1.5rem;'>
            Upload <strong>ANY</strong> customer dataset - no format restrictions!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    v2_col1, v2_col2 = st.columns([3, 2])
    
    with v2_col1:
        st.markdown("### 🎁 Version 2.0 Features:")
        st.markdown("""
        - ✅ **Dynamic Column Mapping** - Map your columns to standard fields
        - ✅ **Flexible Feature Engineering** - Create custom features on the fly
        - ✅ **Universal Data Support** - Works with any CSV structure
        - ✅ **CRM Integration** - Direct export to Salesforce, HubSpot, Mailchimp
        - ✅ **API Access** - Programmatic clustering for your applications
        - ✅ **Save & Load Projects** - Resume work anytime
        - ✅ **Advanced Algorithms** - DBSCAN, HDBSCAN, and more
        - ✅ **Team Collaboration** - Share projects with your team
        - ✅ **White-label Options** - Custom branding for enterprises
        """)
    
    with v2_col2:
        st.markdown("### 📧 Join the Waitlist")
        
        # Waitlist form
        with st.form("waitlist_form"):
            st.markdown("Be the first to know when Version 2.0 launches!")
            
            name = st.text_input("Name *", placeholder="John Doe")
            email = st.text_input("Email *", placeholder="john@company.com")
            company = st.text_input("Company (Optional)", placeholder="Your Company")
            use_case = st.selectbox("Primary Use Case", 
                                    ["Select One", "E-commerce", "Retail", "SaaS", 
                                     "Marketing Agency", "Financial Services", "Other"])
            
            submitted = st.form_submit_button("🚀 Join Waitlist", use_container_width=True)
            
            if submitted:
                if not name or not email:
                    st.error("❌ Please provide your name and email")
                elif "@" not in email:
                    st.error("❌ Please provide a valid email address")
                elif use_case == "Select One":
                    st.error("❌ Please select your primary use case")
                else:
                    # Save to CSV (in production, use database)
                    import csv
                    from datetime import datetime
                    
                    try:
                        # Create waitlist file if doesn't exist
                        waitlist_file = "../results/v2_waitlist.csv"
                        file_exists = False
                        
                        try:
                            with open(waitlist_file, 'r') as f:
                                file_exists = True
                        except FileNotFoundError:
                            pass
                        
                        with open(waitlist_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(['Timestamp', 'Name', 'Email', 'Company', 'Use Case'])
                            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                           name, email, company, use_case])
                        
                        st.success("🎉 Thank you! You've been added to the Version 2.0 waitlist. We'll notify you when it launches!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error saving to waitlist: {str(e)}")
    
    st.markdown("---")
    
    # Comparison Table
    st.markdown("### 📊 Version Comparison")
    
    comparison_data = {
        'Feature': [
            'Upload Data Format',
            'Column Mapping',
            'Feature Engineering',
            'Clustering Algorithms',
            'Data Export',
            'CRM Integration',
            'API Access',
            'Project Save/Load',
            'Pricing'
        ],
        'Version 1.0 (Current)': [
            'Specific format required',
            '❌ Fixed columns',
            '✅ Auto-generated',
            '✅ K-Means, HAC, GMM',
            '✅ CSV download',
            '❌ Manual import',
            '❌ Not available',
            '❌ Session-based',
            '🆓 Free'
        ],
        'Version 2.0 (Coming Soon)': [
            'ANY CSV format',
            '✅ User-defined mapping',
            '✅ Custom + Auto',
            '✅ 7+ algorithms',
            '✅ CSV, JSON, API',
            '✅ Direct integration',
            '✅ REST API',
            '✅ Persistent storage',
            '💰 Freemium + Pro'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Call to Action
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background-color: #1f77b4; border-radius: 10px;'>
            <h3>Ready to get started with Version 1.0?</h3>
            <p>Download the sample data and try it out now!</p>
        </div>
        """, unsafe_allow_html=True)

# UPLOAD DATA PAGE
elif menu == "📤 Upload Data":
    st.markdown("## Upload Your Customer Data")
    
    # Sample data download
    st.markdown("### 📥 Don't have data? Download our sample:")
    try:
        with open("../data/customer_segmentation.csv", "rb") as f:
            st.download_button(
                label="Download Sample CSV",
                data=f.read(),
                file_name="sample_customer_data.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.info("Sample data not available. Please upload your own CSV file.")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            st.success(f"✅ Data loaded successfully! {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Data preview
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{df.shape[0]:,}")
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            col4.metric("Missing Values", df.isnull().sum().sum())
            
            # Data types
            with st.expander("📋 View Column Details"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Missing': df.isnull().sum().values,
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to get started")

# EDA PAGE
elif menu == "📊 EDA":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the 'Upload Data' section")
    else:
        df = st.session_state.df
        st.markdown("## Exploratory Data Analysis")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 Demographics", "🛒 Behavioral", "🔗 Correlations", "🔍 Outliers", "⚠️ Data Quality"])        
        with tab1:
            st.markdown("### Demographic Analysis")
            
            col1, col2 = st.columns(2)
            
            # Age distribution
            with col1:
                if 'Age' in df.columns:
                    fig_age = px.histogram(df, x='Age', nbins=30, 
                                          title="Age Distribution",
                                          labels={'Age': 'Age (years)', 'count': 'Number of Customers'})
                    fig_age.update_layout(bargap=0.1)
                    st.plotly_chart(fig_age, use_container_width=True)
                elif 'Year_Birth' in df.columns:
                    df['Age'] = 2025 - df['Year_Birth']
                    fig_age = px.histogram(df, x='Age', nbins=30, 
                                          title="Age Distribution",
                                          labels={'Age': 'Age (years)', 'count': 'Number of Customers'})
                    fig_age.update_layout(bargap=0.1)
                    st.plotly_chart(fig_age, use_container_width=True)
            
            # Income distribution
            with col2:
                if 'Income' in df.columns:
                    fig_income = px.histogram(df, x='Income', nbins=30,
                                             title="Income Distribution",
                                             labels={'Income': 'Income ($)', 'count': 'Number of Customers'})
                    fig_income.update_layout(bargap=0.1)
                    st.plotly_chart(fig_income, use_container_width=True)
            
            # Education distribution
            col3, col4 = st.columns(2)
            
            with col3:
                if 'Education' in df.columns:
                    edu_counts = df['Education'].value_counts()
                    fig_edu = px.pie(values=edu_counts.values, names=edu_counts.index,
                                    title="Education Distribution")
                    st.plotly_chart(fig_edu, use_container_width=True)
            
            # Marital Status distribution
            with col4:
                if 'Marital_Status' in df.columns:
                    marital_counts = df['Marital_Status'].value_counts()
                    fig_marital = px.pie(values=marital_counts.values, names=marital_counts.index,
                                        title="Marital Status Distribution")
                    st.plotly_chart(fig_marital, use_container_width=True)
        
        with tab2:
            st.markdown("### Behavioural Analysis")
            
            col1, col2 = st.columns(2)
            
            # Spending analysis
            with col1:
                spending_cols = [col for col in df.columns if col.startswith('Mnt')]
                if len(spending_cols) > 0:
                    if 'Total_Spending' not in df.columns:
                        df['Total_Spending'] = df[spending_cols].sum(axis=1)
                    
                    fig_spend = px.histogram(df, x='Total_Spending', nbins=30,
                                            title="Total Spending Distribution",
                                            labels={'Total_Spending': 'Total Spending ($)'})
                    fig_spend.update_layout(bargap=0.1)
                    st.plotly_chart(fig_spend, use_container_width=True)
                    
                    # Spending by category
                    if len(spending_cols) > 0:
                        spend_by_cat = df[spending_cols].sum().sort_values(ascending=False)
                        fig_cat = px.bar(x=spend_by_cat.index, y=spend_by_cat.values,
                                        title="Spending by Product Category",
                                        labels={'x': 'Category', 'y': 'Total Spending ($)'})
                        st.plotly_chart(fig_cat, use_container_width=True)
            
            # Purchase behavior
            with col2:
                purchase_cols = [col for col in df.columns if 'Purchase' in col and 'Num' in col]
                if len(purchase_cols) > 0:
                    if 'Total_Purchases' not in df.columns:
                        df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
                    
                    fig_purch = px.histogram(df, x='Total_Purchases', nbins=30,
                                            title="Total Purchases Distribution",
                                            labels={'Total_Purchases': 'Number of Purchases'})
                    fig_purch.update_layout(bargap=0.1)
                    st.plotly_chart(fig_purch, use_container_width=True)
                    
                    # Purchases by channel
                    if len(purchase_cols) > 0:
                        purch_by_channel = df[purchase_cols].sum().sort_values(ascending=False)
                        fig_channel = px.bar(x=purch_by_channel.index, y=purch_by_channel.values,
                                            title="Purchases by Channel",
                                            labels={'x': 'Channel', 'y': 'Total Purchases'})
                        st.plotly_chart(fig_channel, use_container_width=True)
            
            # Recency and Web Visits
            col3, col4 = st.columns(2)
            
            with col3:
                if 'Recency' in df.columns:
                    fig_rec = px.histogram(df, x='Recency', nbins=30,
                                          title="Recency Distribution (Days Since Last Purchase)",
                                          labels={'Recency': 'Days'})
                    fig_rec.update_layout(bargap=0.1)
                    st.plotly_chart(fig_rec, use_container_width=True)
            
            with col4:
                if 'NumWebVisitsMonth' in df.columns:
                    fig_web = px.histogram(df, x='NumWebVisitsMonth', nbins=20,
                                          title="Web Visits per Month",
                                          labels={'NumWebVisitsMonth': 'Number of Visits'})
                    fig_web.update_layout(bargap=0.1)
                    st.plotly_chart(fig_web, use_container_width=True)
        
        with tab3:
            st.markdown("### Correlation Analysis")
            
            # Select key behavioral features for correlation
            corr_features = []
            for col in ['Age', 'Income', 'Total_Spending', 'Total_Purchases', 'Recency', 'NumWebVisitsMonth']:
                if col in df.columns:
                    corr_features.append(col)
            
            if len(corr_features) > 1:
                corr_matrix = df[corr_features].corr()
                
                fig_corr = px.imshow(corr_matrix, 
                                    text_auto='.2f',
                                    aspect="auto",
                                    color_continuous_scale='RdBu_r',
                                    title="Feature Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Key insights with interpretation
                st.markdown("#### 🔍 Correlation Insights")
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False).head(5)
                st.dataframe(corr_df.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
                
                # Interpret top correlations
                st.markdown("**📖 What These Correlations Mean:**")
                for idx, row in corr_df.head(3).iterrows():
                    corr_val = row['Correlation']
                    feat1 = row['Feature 1']
                    feat2 = row['Feature 2']
                    
                    if abs(corr_val) > 0.7:
                        strength = "**Strong**"
                    elif abs(corr_val) > 0.4:
                        strength = "**Moderate**"
                    else:
                        strength = "**Weak**"
                    
                    direction = "positive" if corr_val > 0 else "negative"
                    
                    if corr_val > 0:
                        interpretation = f"{strength} {direction} correlation ({corr_val:.2f}): As {feat1} increases, {feat2} tends to increase as well."
                    else:
                        interpretation = f"{strength} {direction} correlation ({corr_val:.2f}): As {feat1} increases, {feat2} tends to decrease."
                    
                    st.write(f"• {interpretation}")
            else:
                st.info("Need at least 2 numeric features for correlation analysis")
        
        with tab4:
            st.markdown("### Outlier Analysis")
            
            st.warning("⚠️ **Important**: Not all outliers are errors. High spending values often represent premium/VIP customers who are valuable to your business. Only remove outliers that are clear data entry errors (e.g., impossible ages, income values with suspicious patterns like repeating digits, or extreme values that don't align with your typical customer demographics). Review each outlier in context before deciding to remove it.")
            
            st.markdown("---")
            
            
            # Select features to analyze
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if 'id' not in col.lower() and 'ID' not in col]
            
            if len(numeric_cols) > 0:
                selected_outlier_features = st.multiselect(
                    "Select features to analyze for outliers:",
                    numeric_cols,
                    default=[col for col in ['Age', 'Income', 'Total_Spending'] if col in numeric_cols]
                )
                
                if len(selected_outlier_features) > 0:
                    # Calculate outliers with detailed analysis
                    for feature in selected_outlier_features:
                        st.markdown(f"#### 🔍 {feature} Analysis")
                        
                        Q1 = df[feature].quantile(0.25)
                        Q3 = df[feature].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outlier_df = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
                        outlier_count = len(outlier_df)
                        outlier_pct = (outlier_count / len(df)) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Outliers", outlier_count)
                        col2.metric("Percentage", f"{outlier_pct:.2f}%")
                        col3.metric("Valid Range", f"[{lower_bound:.0f}, {upper_bound:.0f}]")
                        col4.metric("Actual Range", f"[{df[feature].min():.0f}, {df[feature].max():.0f}]")
                        
                        # Show actual outlier values
                        if outlier_count > 0 and outlier_count <= 50:
                            with st.expander(f"View {outlier_count} outlier values"):
                                outlier_values = outlier_df[feature].sort_values(ascending=False).values
                                
                                # Interpret outliers
                                if feature == 'Age' and any(outlier_values > 100):
                                    st.warning("⚠️ Ages above 100 detected - likely data entry errors")
                                elif feature == 'Income' and any(outlier_values > 200000):
                                    st.warning("⚠️ Very high incomes detected - check for data errors (e.g., 666666)")
                                elif feature in ['Total_Spending', 'MntWines', 'MntMeatProducts']:
                                    st.info("ℹ️ High spending values may represent legitimate VIP/premium customers")
                                
                                # Show outlier table
                                outlier_display = pd.DataFrame({
                                    'Value': outlier_values,
                                    'Type': ['Above upper bound' if v > upper_bound else 'Below lower bound' 
                                            for v in outlier_values]
                                })
                                st.dataframe(outlier_display, use_container_width=True)
                        
                        elif outlier_count > 50:
                            st.info(f"Too many outliers ({outlier_count}) to display. This suggests these may be legitimate high-value customers rather than data errors.")
                        
                        # Boxplot
                        fig_box = px.box(df, y=feature, title=f"{feature} Distribution with Outliers")
                        st.plotly_chart(fig_box, use_container_width=True)
                        
                        st.markdown("---")                    
                    # Create grid of boxplots
                    n_features = len(selected_outlier_features)
                    n_cols = 3
                    n_rows = (n_features + n_cols - 1) // n_cols
                    
                    for i in range(0, n_features, n_cols):
                        cols = st.columns(n_cols)
                        for j, col in enumerate(cols):
                            if i + j < n_features:
                                feature = selected_outlier_features[i + j]
                                with col:
                                    fig_box = px.box(df, y=feature, title=f"{feature}")
                                    st.plotly_chart(fig_box, use_container_width=True)

        with tab5:
            st.markdown("### Data Quality Checks")
            
            # Duplicates
            st.markdown("#### 🔄 Duplicate Records")
            
            # Check for exact duplicates across ALL columns
            exact_duplicates = df[df.duplicated(keep=False)]
            exact_dup_count = len(exact_duplicates)
            
            col1, col2 = st.columns(2)
            col1.metric("Exact Duplicate Rows", exact_dup_count)
            col2.metric("Percentage", f"{(exact_dup_count/len(df)*100):.2f}%")
            
            if exact_dup_count > 0:
                st.warning(f"⚠️ Found {exact_dup_count} exact duplicate records (all columns identical). These may be data entry errors.")
                
                with st.expander("View Exact Duplicate Records"):
                    st.dataframe(exact_duplicates, use_container_width=True)
            else:
                st.success("✅ No exact duplicate records found (across all columns)")
            
            # Check for duplicates on key columns only
            key_cols = [col for col in ['Age', 'Income', 'Total_Spending', 'Total_Purchases'] if col in df.columns]
            if len(key_cols) >= 2:
                partial_duplicates = df[df.duplicated(subset=key_cols, keep=False)]
                partial_dup_count = len(partial_duplicates)
                
                if partial_dup_count > exact_dup_count:
                    st.info(f"ℹ️ Found {partial_dup_count} records with duplicate key features ({', '.join(key_cols)}). These may represent similar customers or data patterns.")
                    
                    with st.expander(f"View Partial Duplicates (matching on {', '.join(key_cols)})"):
                        st.dataframe(partial_duplicates[key_cols + ['Education', 'Marital_Status'] if 'Education' in df.columns else key_cols].head(20), 
                                   use_container_width=True)
            
            st.info("💡 During preprocessing, duplicates will be checked on your selected clustering features only.")
            
            st.markdown("---")
            
            # Constant Features (no variance)
            st.markdown("#### 📏 Constant Features (Zero Variance)")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            constant_features = []
            
            for col in numeric_cols:
                if df[col].std() == 0 or df[col].nunique() == 1:
                    constant_features.append(col)
            
            if len(constant_features) > 0:
                st.warning(f"⚠️ Found {len(constant_features)} constant feature(s): {', '.join(constant_features)}")
                st.write("These columns have no variation and should be removed before clustering.")
                
                const_df = pd.DataFrame({
                    'Feature': constant_features,
                    'Value': [df[col].iloc[0] for col in constant_features]
                })
                st.dataframe(const_df, use_container_width=True)
            else:
                st.success("✅ No constant features found")
            
            st.markdown("---")
            
            # Suspicious Values
            st.markdown("#### 🚨 Suspicious Values")
            
            issues = []
            
            # Check for negative income/spending
            if 'Income' in df.columns and (df['Income'] < 0).any():
                neg_income = (df['Income'] < 0).sum()
                issues.append(f"• {neg_income} negative Income values")
            
            spending_cols = [col for col in df.columns if col.startswith('Mnt')]
            for col in spending_cols:
                if (df[col] < 0).any():
                    neg_spend = (df[col] < 0).sum()
                    issues.append(f"• {neg_spend} negative values in {col}")
            
            # Check for placeholder values
            if 'Income' in df.columns:
                placeholder_patterns = [666666, 999999, 0, -1]
                for pattern in placeholder_patterns:
                    count = (df['Income'] == pattern).sum()
                    if count > 0:
                        issues.append(f"• {count} suspicious Income values ({pattern})")
            
            if len(issues) > 0:
                st.warning("⚠️ Suspicious values detected:")
                for issue in issues:
                    st.write(issue)
                st.info("Review these values - they may be data entry errors or placeholders.")
            else:
                st.success("✅ No obvious suspicious values detected")
            
            st.markdown("---")
            
            # Missing Value Summary
            st.markdown("#### 📋 Missing Values Summary")
            missing_summary = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_summary = missing_summary[missing_summary['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if len(missing_summary) > 0:
                st.dataframe(missing_summary, use_container_width=True)
            else:
                st.success("✅ No missing values found")
                            


# Clustering, Download, and Glossary

# CLUSTERING PAGE
elif menu == "🎯 Clustering":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the 'Upload Data' section")
    else:
        df = st.session_state.df.copy()
        st.markdown("## Customer Segmentation Clustering")
        
        # Step 1: Feature Engineering
        st.markdown("### Step 1: Feature Engineering")
        with st.expander("Configure Feature Engineering", expanded=True):
            # Check for necessary columns
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Year_Birth' in df.columns:
                    current_year = st.number_input("Current Year", value=2025, min_value=2020, max_value=2030)
                    df['Age'] = current_year - df['Year_Birth']
                    st.success("✅ Age feature created")
            
            with col2:
                # Total Spending
                spending_cols = [col for col in df.columns if col.startswith('Mnt')]
                if len(spending_cols) > 0:
                    df['Total_Spending'] = df[spending_cols].sum(axis=1)
                    st.success(f"✅ Total_Spending created from {len(spending_cols)} columns")
            
            # Total Purchases
            purchase_cols = [col for col in df.columns if 'Purchase' in col]
            if len(purchase_cols) > 0:
                df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
                st.success(f"✅ Total_Purchases created")
            
            # Total Children
            if 'Kidhome' in df.columns and 'Teenhome' in df.columns:
                df['Total_Children'] = df['Kidhome'] + df['Teenhome']
                st.success("✅ Total_Children created")
        
        st.markdown("---")
        
        # Step 2: Feature Selection
        st.markdown("### Step 2: Select Features for Clustering")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID columns
        numeric_cols = [col for col in numeric_cols if 'id' not in col.lower() and 'ID' not in col]
        
        selected_features = st.multiselect(
            "Choose features for clustering:",
            numeric_cols,
            default=[col for col in ['Age', 'Income', 'Total_Spending', 'Total_Purchases'] if col in numeric_cols]
        )
        
        if len(selected_features) < 2:
            st.error("⚠️ Please select at least 2 features for clustering")
            st.stop()
        
        st.info(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
        
        st.markdown("---")
        
        # Step 3: Preprocessing
        st.markdown("### Step 3: Data Preprocessing")
        
        st.warning("⚠️ **Important**: Not all outliers are errors. High spending values often represent premium/VIP customers who are valuable to your business. Only remove outliers that are clear data entry errors (e.g., impossible ages, income values with suspicious patterns like repeating digits, or extreme values that don't align with your typical customer demographics). Review each outlier in context before deciding to remove it.")
        
        st.markdown("---")
        
        # Data Quality Options
        st.markdown("#### 🔧 Data Quality Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            handle_missing = st.selectbox("Handle Missing Values:", 
                                         ["Select Option", "Fill with Median", "Fill with Mean", "Drop Rows"],
                                         help="Choose how to handle missing values in your data")
        
        with col2:
            remove_duplicates = st.checkbox("Remove Duplicate Rows", value=False,
                                           help="Remove exact duplicate records that may be data entry errors")
        
        with col3:
            remove_outliers = st.checkbox("Remove Outliers", value=False)
        
        # Validation: Ensure missing value method is selected
        if handle_missing == "Select Option":
            st.error("⚠️ Please select a method for handling missing values before proceeding")
            st.stop()
        
        # If removing outliers, let user select which features
        outlier_features_to_clean = []
        if remove_outliers:
            outlier_features_to_clean = st.multiselect(
                "Select features to remove outliers from:",
                selected_features,
                default=[feat for feat in ['Age', 'Income'] if feat in selected_features],
                help="Choose features where outliers are likely data errors. Keep spending outliers (they're often VIP customers)."
            )
            
            if len(outlier_features_to_clean) == 0:
                st.warning("⚠️ Please select at least one feature or uncheck 'Remove Outliers'")
        
        st.markdown("---")
        
        # Process data - keep track of original indices
        df_cluster = df[selected_features].copy()
        original_indices = df_cluster.index
        preprocessing_steps = []
        
        # Step 1: Remove duplicates if selected
        if remove_duplicates:
            rows_before = len(df_cluster)
            df_cluster = df_cluster.drop_duplicates()
            original_indices = df_cluster.index
            rows_removed = rows_before - len(df_cluster)
            if rows_removed > 0:
                preprocessing_steps.append(f"Removed {rows_removed} duplicate rows")
        
        # Step 2: Remove constant features (auto)
        constant_features = []
        for col in df_cluster.columns:
            if df_cluster[col].std() == 0 or df_cluster[col].nunique() == 1:
                constant_features.append(col)
        
        if len(constant_features) > 0:
            df_cluster = df_cluster.drop(columns=constant_features)
            preprocessing_steps.append(f"⚠️ Auto-removed {len(constant_features)} constant feature(s): {', '.join(constant_features)}")
            st.warning(f"Constant features removed: {', '.join(constant_features)} (no variance)")
        
        # Step 3: Handle missing values
        missing_count = df_cluster.isnull().sum().sum()
        
        if missing_count > 0:
            if handle_missing == "Fill with Median":
                df_cluster = df_cluster.fillna(df_cluster.median())
                preprocessing_steps.append(f"Filled {missing_count} missing values with median")
            elif handle_missing == "Fill with Mean":
                df_cluster = df_cluster.fillna(df_cluster.mean())
                preprocessing_steps.append(f"Filled {missing_count} missing values with mean")
            else:  # Drop Rows
                rows_before = len(df_cluster)
                df_cluster = df_cluster.dropna()
                original_indices = df_cluster.index
                rows_removed = rows_before - len(df_cluster)
                if rows_removed > 0:
                    preprocessing_steps.append(f"Dropped {rows_removed} rows with missing values")
        else:
            preprocessing_steps.append("No missing values found")
        
        # Step 4: Remove outliers from selected features only
        if remove_outliers and len(outlier_features_to_clean) > 0:
            outlier_mask = pd.Series(True, index=df_cluster.index)
            
            for col in outlier_features_to_clean:
                Q1 = df_cluster[col].quantile(0.25)
                Q3 = df_cluster[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_mask &= (df_cluster[col] >= lower) & (df_cluster[col] <= upper)
            
            rows_before = len(df_cluster)
            df_cluster = df_cluster[outlier_mask]
            original_indices = df_cluster.index
            rows_removed = rows_before - len(df_cluster)
            preprocessing_steps.append(f"Removed {rows_removed} outlier rows from {', '.join(outlier_features_to_clean)}")
        
        # Show preprocessing summary
        if len(preprocessing_steps) > 0:
            st.info("**Preprocessing Steps Applied:**\n" + "\n".join([f"• {step}" for step in preprocessing_steps]))
        
        # Check if we have enough data left
        if len(df_cluster) < 10:
            st.error("❌ Too few rows remaining after preprocessing (< 10). Please adjust your settings.")
            st.stop()
        
        # Check if we have enough features
        if len(df_cluster.columns) < 2:
            st.error("❌ Need at least 2 features for clustering. All features were removed (likely constant features).")
            st.stop()
        
        # Scale features
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_cluster)
        
        st.success(f"✅ Data preprocessed: {df_cluster.shape[0]} rows × {df_cluster.shape[1]} features ready for clustering")
        
        st.markdown("---")
        
        # Step 4: Algorithm Selection
        st.markdown("### Step 4: Select Clustering Algorithm")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            algorithm = st.selectbox("Choose Algorithm:", 
                                    ["K-Means", "Hierarchical (HAC)", "Gaussian Mixture Model (GMM)"])
            
            n_clusters = st.slider("Number of Clusters:", min_value=2, max_value=10, value=3)
        
        with col2:
            st.markdown("**Algorithm Descriptions:**")
            if algorithm == "K-Means":
                st.info("🔵 **K-Means**: Partition-based clustering. Fast and efficient for large datasets. Assumes spherical clusters.")
            elif algorithm == "Hierarchical (HAC)":
                st.info("🟢 **Hierarchical**: Builds a tree of clusters. Good for understanding hierarchical relationships.")
            else:
                st.info("🟠 **GMM**: Probabilistic clustering. Handles overlapping clusters and irregular shapes. Best for continuous data.")
        
        # Run Clustering Button
        if st.button("🚀 Run Clustering", type="primary"):
            with st.spinner("Running clustering algorithm..."):
                
                # Apply selected algorithm
                if algorithm == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = model.fit_predict(df_scaled)
                elif algorithm == "Hierarchical (HAC)":
                    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    labels = model.fit_predict(df_scaled)
                else:  # GMM
                    model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
                    labels = model.fit_predict(df_scaled)
                
                # Calculate metrics
                silhouette = silhouette_score(df_scaled, labels)
                davies_bouldin = davies_bouldin_score(df_scaled, labels)
                calinski = calinski_harabasz_score(df_scaled, labels)
                
                # Add labels to dataframe - ONLY to rows that were clustered
                df_filtered = df.loc[original_indices].copy()
                df_filtered['Cluster'] = labels
                
                # Store properly in session state
                st.session_state.df_filtered = df_filtered
                st.session_state.df_scaled = df_scaled
                st.session_state.labels = labels
                st.session_state.original_indices = original_indices
                st.session_state.clusters_computed = True
                st.session_state.algorithm = algorithm
                st.session_state.selected_features = selected_features
                
                st.success("✅ Clustering complete!")
                
                # Display metrics
                st.markdown("### 📊 Clustering Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Clusters", n_clusters)
                col2.metric("Silhouette Score", f"{silhouette:.3f}")
                col3.metric("Davies-Bouldin", f"{davies_bouldin:.3f}")
                col4.metric("Calinski-Harabasz", f"{calinski:.2f}")
                
                st.info(f"ℹ️ Metrics calculated on {len(df_filtered)} customers after preprocessing")
                

                # Display metrics
                st.markdown("### 📊 Clustering Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Clusters", n_clusters)
                col2.metric("Silhouette Score", f"{silhouette:.3f}")
                col3.metric("Davies-Bouldin", f"{davies_bouldin:.3f}")
                col4.metric("Calinski-Harabasz", f"{calinski:.2f}")
                
                st.info(f"ℹ️ Metrics calculated on {len(df_filtered)} customers after preprocessing")
                
                # ADD CLUSTER QUALITY INTERPRETATION HERE
                st.markdown("---")
                st.markdown("### 🎯 Cluster Quality Assessment")
                
                # Interpret Silhouette Score
                if silhouette > 0.7:
                    sil_assessment = "**Excellent** ✅ - Clusters are well-separated and distinct"
                    sil_color = "green"
                elif silhouette > 0.5:
                    sil_assessment = "**Good** ✅ - Clusters are reasonably separated"
                    sil_color = "green"
                elif silhouette > 0.3:
                    sil_assessment = "**Fair** ⚠️ - Clusters overlap somewhat but are distinguishable"
                    sil_color = "orange"
                else:
                    sil_assessment = "**Poor** ❌ - Clusters overlap significantly"
                    sil_color = "red"
                
                # Interpret Davies-Bouldin
                if davies_bouldin < 1.0:
                    db_assessment = "**Excellent** ✅ - Very distinct clusters"
                    db_color = "green"
                elif davies_bouldin < 2.0:
                    db_assessment = "**Good** ✅ - Well-separated clusters"
                    db_color = "green"
                elif davies_bouldin < 3.0:
                    db_assessment = "**Fair** ⚠️ - Moderate separation"
                    db_color = "orange"
                else:
                    db_assessment = "**Poor** ❌ - Clusters not well-separated"
                    db_color = "red"
                
                # Interpret Calinski-Harabasz
                if calinski > 1000:
                    ch_assessment = "**Excellent** ✅ - Strong cluster structure"
                    ch_color = "green"
                elif calinski > 500:
                    ch_assessment = "**Good** ✅ - Clear cluster structure"
                    ch_color = "green"
                elif calinski > 200:
                    ch_assessment = "**Fair** ⚠️ - Moderate cluster structure"
                    ch_color = "orange"
                else:
                    ch_assessment = "**Poor** ❌ - Weak cluster structure"
                    ch_color = "red"
                
                # Display interpretations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Silhouette Score**")
                    st.markdown(f"{sil_assessment}")
                    with st.expander("What does this mean?"):
                        st.write(f"""
                        **Your score: {silhouette:.3f}**
                        
                        Silhouette measures how similar customers are to their own cluster compared to other clusters.
                        
                        - **Range:** -1 to 1
                        - **Your result:** {sil_assessment.split()[0]} separation
                        - **Interpretation:** {'Customers in each cluster are distinct from other clusters.' if silhouette > 0.5 else 'Some overlap between clusters exists, which is normal for customer behaviour data.'}
                        """)
                
                with col2:
                    st.markdown("**Davies-Bouldin Index**")
                    st.markdown(f"{db_assessment}")
                    with st.expander("What does this mean?"):
                        st.write(f"""
                        **Your score: {davies_bouldin:.3f}**
                        
                        Davies-Bouldin measures cluster compactness and separation.
                        
                        - **Range:** 0 to ∞ (lower is better)
                        - **Your result:** {db_assessment.split()[0]} quality
                        - **Interpretation:** {'Clusters are compact and well-separated.' if davies_bouldin < 2.0 else 'Clusters have moderate separation, which is acceptable for behavioural segmentation.'}
                        """)
                
                with col3:
                    st.markdown("**Calinski-Harabasz Score**")
                    st.markdown(f"{ch_assessment}")
                    with st.expander("What does this mean?"):
                        st.write(f"""
                        **Your score: {calinski:.2f}**
                        
                        Calinski-Harabasz measures the ratio of between-cluster to within-cluster variance.
                        
                        - **Range:** 0 to ∞ (higher is better)
                        - **Your result:** {ch_assessment.split()[0]} structure
                        - **Interpretation:** {'Strong evidence of distinct cluster structure.' if calinski > 500 else 'Moderate cluster structure detected.'}
                        """)
                
                # Overall assessment
                st.markdown("---")
                
                # Calculate overall score (simple average of normalized metrics)
                sil_norm = max(0, min(1, (silhouette + 1) / 2))  # Normalize -1 to 1 → 0 to 1
                db_norm = max(0, min(1, 1 - (davies_bouldin / 5)))  # Lower is better, cap at 5
                ch_norm = max(0, min(1, calinski / 2000))  # Normalize to 0-1, cap at 2000
                
                overall_score = (sil_norm + db_norm + ch_norm) / 3
                
                if overall_score > 0.6:
                    st.success(f"""
                    ✅ **Overall Assessment: Good Clustering Quality**
                    
                    Your {n_clusters} clusters are meaningful and suitable for customer segmentation. The clusters show reasonable separation 
                    and can be used for targeted marketing strategies. This is typical for real-world customer behaviour data, 
                    where perfect separation is rare due to the complexity of human behaviour.
                    """)
                elif overall_score > 0.4:
                    st.warning(f"""
                    ⚠️ **Overall Assessment: Fair Clustering Quality**
                    
                    Your {n_clusters} clusters show moderate separation. They can still be used for segmentation, but consider:
                    - Trying a different number of clusters
                    - Selecting different features
                    - Testing another algorithm (try GMM if using K-Means, or vice versa)
                    """)
                else:
                    st.error(f"""
                    ❌ **Overall Assessment: Weak Clustering Quality**
                    
                    The current clustering shows poor separation. Recommendations:
                    - Try different feature combinations
                    - Adjust the number of clusters (2-5 typically work best)
                    - Consider if your data has natural groupings (some datasets don't cluster well)
                    - Remove more outliers or try different preprocessing
                    """)
                
                st.markdown("---")


                # Visualizations
                st.markdown("### 📈 Cluster Visualizations")
                
                # PCA for 2D visualization
                pca = PCA(n_components=2, random_state=42)
                df_pca = pca.fit_transform(df_scaled)
                
                plot_df = pd.DataFrame({
                    'PC1': df_pca[:, 0],
                    'PC2': df_pca[:, 1],
                    'Cluster': labels
                })
                
                fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                               title=f'{algorithm} Clusters (2D PCA Projection)',
                               color_continuous_scale='viridis',
                               labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                                      'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster distribution - FIX THE ERROR
                st.markdown("### 📊 Cluster Distribution")
                cluster_counts = df_filtered['Cluster'].value_counts().sort_index()
                cluster_df = pd.DataFrame({
                    'Cluster': cluster_counts.index,
                    'Count': cluster_counts.values,
                    'Percentage': (cluster_counts.values / len(df_filtered) * 100).round(2)
                })
                
                fig_dist = px.bar(cluster_df, x='Cluster', y='Count', 
                                 text='Percentage',
                                 title="Customers per Cluster",
                                 labels={'Count': 'Number of Customers'})
                fig_dist.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_dist.update_layout(bargap=0.2)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Cluster profiles
                st.markdown("### 📋 Cluster Profiles")
                profile = df_filtered.groupby('Cluster')[selected_features].mean().round(2)
                st.dataframe(profile, use_container_width=True)
                
                # Heatmap
                fig_heat = px.imshow(profile.T, 
                                    text_auto='.0f',
                                    aspect="auto",
                                    color_continuous_scale='RdYlGn',
                                    title="Cluster Profile Heatmap")
                st.plotly_chart(fig_heat, use_container_width=True)

                
                # CLUSTER INTERPRETATION & BUSINESS INSIGHTS
                st.markdown("---")
                st.markdown("### 💼 Cluster Interpretation & Business Insights")
                
                # Analyze each cluster
                for cluster_id in sorted(df_filtered['Cluster'].unique()):
                    cluster_data = df_filtered[df_filtered['Cluster'] == cluster_id]
                    cluster_size = len(cluster_data)
                    cluster_pct = (cluster_size / len(df_filtered)) * 100
                    
                    with st.expander(f"**Cluster {cluster_id}** - {cluster_size} customers ({cluster_pct:.1f}%)", expanded=True):
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        if 'Age' in selected_features:
                            col1.metric("Avg Age", f"{cluster_data['Age'].mean():.0f} years")
                        if 'Income' in selected_features:
                            col2.metric("Avg Income", f"${cluster_data['Income'].mean():,.0f}")
                        if 'Total_Spending' in selected_features:
                            col3.metric("Avg Spending", f"${cluster_data['Total_Spending'].mean():,.0f}")
                        if 'Total_Purchases' in selected_features:
                            col4.metric("Avg Purchases", f"{cluster_data['Total_Purchases'].mean():.1f}")
                        
                        st.markdown("---")
                        
                        # Profile interpretation
                        avg_income = cluster_data['Income'].mean() if 'Income' in selected_features else 0
                        avg_spending = cluster_data['Total_Spending'].mean() if 'Total_Spending' in selected_features else 0
                        
                        # Simple segment classification
                        if avg_spending > df_filtered['Total_Spending'].quantile(0.75):
                            segment_name = "Premium Customers"
                            st.markdown("#### 🌟 Premium Customers")
                            st.markdown("""
                            **Profile:** High-value customers with significant spending power
                            
                            **Characteristics:**
                            - Highest income and spending levels
                            - Quality-focused, brand loyal
                            - Low price sensitivity
                            
                            **Marketing Strategy:**
                            - VIP programs and exclusive offers
                            - Premium product lines
                            - Personalized concierge service
                            - Early access to new products
                            - Invitation-only events
                            """)
                        elif avg_spending > df_filtered['Total_Spending'].quantile(0.33):
                            segment_name = "Standard Customers"
                            st.markdown("#### 💼 Standard Customers")
                            st.markdown("""
                            **Profile:** Core customer base with moderate spending
                            
                            **Characteristics:**
                            - Middle-income, consistent buyers
                            - Value-conscious but quality-aware
                            - Regular engagement
                            
                            **Marketing Strategy:**
                            - Cross-selling opportunities
                            - Seasonal campaigns and promotions
                            - Loyalty rewards programs
                            - Bundle offers
                            - Retention-focused initiatives
                            """)
                        else:
                            segment_name = "Budget Shoppers"
                            st.markdown("#### 🛒 Budget Shoppers")
                            st.markdown("""
                            **Profile:** Price-sensitive customers with limited spending
                            
                            **Characteristics:**
                            - Lower income, budget-conscious
                            - High research/browsing activity
                            - Deal-seekers, price-sensitive
                            
                            **Marketing Strategy:**
                            - Discount programs and clearance alerts
                            - Family bundles and value packs
                            - Loyalty points for repeat purchases
                            - Affordable product lines
                            - High-frequency digital touchpoints
                            """)                

# DOWNLOAD RESULTS PAGE
elif menu == "📥 Download Results":
    if not st.session_state.clusters_computed:
        st.warning("⚠️ Please run clustering first in the 'Clustering' section")
    else:
        df_filtered = st.session_state.df_filtered
        st.markdown("## Download Segmented Results")
        
        st.markdown("### 📊 Results Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df_filtered))
        col2.metric("Number of Clusters", df_filtered['Cluster'].nunique())
        col3.metric("Algorithm Used", st.session_state.algorithm)
        
        st.markdown("---")
        
        # Download all data
        st.markdown("### 📥 Download Complete Dataset with Clusters")
        csv_all = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download All Customers (CSV)",
            data=csv_all,
            file_name="clustered_customers_all.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Download individual segments
        st.markdown("### 📥 Download Individual Segments")
        st.info("💡 Use these files to create targeted marketing campaigns for each customer segment")
        
        for cluster in sorted(df_filtered['Cluster'].unique()):
            cluster_df = df_filtered[df_filtered['Cluster'] == cluster]
            cluster_size = len(cluster_df)
            cluster_pct = (cluster_size / len(df_filtered) * 100)
            
            with st.expander(f"Cluster {cluster} - {cluster_size} customers ({cluster_pct:.1f}%)"):
                # Show preview
                st.dataframe(cluster_df.head(), use_container_width=True)
                
                # Download button
                csv_segment = cluster_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📄 Download Cluster {cluster} (CSV)",
                    data=csv_segment,
                    file_name=f"cluster_{cluster}_customers.csv",
                    mime="text/csv",
                    key=f"download_cluster_{cluster}"
                )
        
        st.markdown("---")
        
        # Marketing recommendations
        st.markdown("### 💡 How to Use These Segments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Email Marketing Platforms:**")
            st.write("- Mailchimp")
            st.write("- Constant Contact")
            st.write("- SendGrid")
            st.write("- HubSpot")
        
        with col2:
            st.markdown("**CRM Systems:**")
            st.write("- Salesforce")
            st.write("- Zoho CRM")
            st.write("- Microsoft Dynamics")
            st.write("- Pipedrive")
        
        st.info("📌 Import the downloaded CSV files into your marketing platform to create targeted campaigns for each customer segment.")


# GLOSSARY PAGE
elif menu == "📖 Glossary":
    st.markdown("## Glossary of Terms")
    
    st.markdown("### 🎯 Clustering Concepts")
    
    with st.expander("**Customer Segmentation**"):
        st.write("""
        The process of dividing customers into distinct groups based on shared characteristics, 
        behaviours, or preferences. Enables targeted marketing and personalized experiences.
        """)
    
    with st.expander("**K-Means Clustering**"):
        st.write("""
        A partition-based clustering algorithm that divides data into K clusters by minimizing 
        the within-cluster sum of squares. Fast and efficient for large datasets. Assumes 
        spherical, evenly-sized clusters.
        
        **Best for:** Large datasets with well-separated, spherical clusters
        """)
    
    with st.expander("**Hierarchical Clustering (HAC)**"):
        st.write("""
        An algorithm that builds a tree (dendrogram) of clusters by iteratively merging or 
        splitting groups. Useful for understanding hierarchical relationships in data.
        
        **Best for:** Understanding cluster hierarchy, smaller datasets
        """)
    
    with st.expander("**Gaussian Mixture Model (GMM)**"):
        st.write("""
        A probabilistic clustering method that models clusters as Gaussian distributions. 
        Can handle overlapping clusters and irregular shapes. Provides soft cluster assignments 
        (probabilities rather than hard labels).
        
        **Best for:** Continuous data, overlapping clusters, probabilistic assignments
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Evaluation Metrics")
    
    with st.expander("**Silhouette Score**"):
        st.write("""
        Measures how similar a point is to its own cluster compared to other clusters.
        
        **Range:** -1 to 1
        - **1**: Perfect clustering
        - **0**: Overlapping clusters
        - **-1**: Incorrect clustering
        
        **Interpretation:** Higher is better (>0.5 is good, >0.7 is excellent)
        """)
    
    with st.expander("**Davies-Bouldin Index**"):
        st.write("""
        Measures the average similarity between each cluster and its most similar cluster. 
        Ratio of within-cluster distances to between-cluster distances.
        
        **Range:** 0 to ∞
        
        **Interpretation:** Lower is better (closer to 0 means better separation)
        """)
    
    with st.expander("**Calinski-Harabasz Score**"):
        st.write("""
        Also known as Variance Ratio Criterion. Measures the ratio of between-cluster 
        dispersion to within-cluster dispersion.
        
        **Range:** 0 to ∞
        
        **Interpretation:** Higher is better (more distinct, well-separated clusters)
        """)
    
    with st.expander("**Inertia (Within-Cluster Sum of Squares)**"):
        st.write("""
        Sum of squared distances of samples to their closest cluster center. Used in the 
        Elbow Method to find optimal K.
        
        **Interpretation:** Lower is better, but decreases with more clusters. Look for the "elbow" 
        where adding clusters gives diminishing returns.
        """)
    
    st.markdown("---")
    st.markdown("### 🔧 Preprocessing Terms")
    
    with st.expander("**Feature Engineering**"):
        st.write("""
        Creating new features from existing data to better capture patterns. Examples:
        - **Age** from Year_Birth
        - **Total_Spending** from individual product spending
        - **Total_Purchases** from different purchase channels
        """)
    
    with st.expander("**Standardization (Scaling)**"):
        st.write("""
        Transforming features to have mean=0 and standard deviation=1. Essential for clustering 
        because algorithms are sensitive to feature scales.
        
        **Formula:** (x - mean) / std_deviation
        
        **Why needed:** Prevents features with large values from dominating the clustering
        """)
    
    with st.expander("**Outliers**"):
        st.write("""
        Data points that significantly differ from other observations. Can be:
        - **Data errors** (age = 131 years)
        - **Legitimate extreme values** (VIP customers with high spending)
        
        **IQR Method:** Points beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR are considered outliers
        """)
    
    with st.expander("**PCA (Principal Component Analysis)**"):
        st.write("""
        Dimensionality reduction technique that transforms many features into fewer components 
        while retaining most information. Used for visualizing high-dimensional clusters in 2D/3D.
        
        **PC1, PC2:** First two principal components capturing the most variance
        """)
    
    st.markdown("---")
    st.markdown("### 💼 Business Terms")
    
    with st.expander("**Customer Lifetime Value (CLV)**"):
        st.write("""
        Total revenue a business expects from a customer over their entire relationship. 
        Premium segments typically have higher CLV.
        """)
    
    with st.expander("**Conversion Rate**"):
        st.write("""
        Percentage of potential customers who complete a desired action (purchase, signup, etc.).
        
        **Formula:** (Conversions / Total Visitors) × 100
        """)
    
    with st.expander("**Churn**"):
        st.write("""
        Rate at which customers stop doing business with a company. Segmentation helps 
        identify at-risk customers for retention campaigns.
        """)
    
    with st.expander("**ROI (Return on Investment)**"):
        st.write("""
        Measure of profitability of marketing campaigns.
        
        **Formula:** (Revenue - Cost) / Cost × 100
        
        Targeted campaigns based on segmentation typically yield higher ROI.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Customer Segmentation Dashboard | Built with Streamlit | © 2025</p>
    <p>Developed by <a href='https://portfolio.edoborosasere.com/' target='_blank' style='color: #1f77b4; text-decoration: none;'>Osasere Edobor</a></p>
</div>
""", unsafe_allow_html=True)