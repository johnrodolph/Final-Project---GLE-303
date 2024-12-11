import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
 
# Load and cache the dataset
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
 
# Prepare data for analysis
def clean_data(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove 'Unnamed' columns
    return df.dropna()  # Remove missing values
 
# Display basic descriptive statistics
def describe_data(df):
    return df.describe()
 
# Application layout
st.title("Data Analysis Application")
 
# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
 
if uploaded_file is not None:
    data = load_data(uploaded_file)
 
    if data is not None:
        # Sidebar for navigation
        with st.sidebar:
            st.header("Navigation")
            selected_tab = st.radio("Go to:", ["Overview", "Data Cleaning", "Visualization", "K-means Clustering", "Linear Regression", "Conclusions"])
 
        if selected_tab == "Overview":
            st.header("Overview")
            st.write("Dataset Preview:")
            st.dataframe(data.head())
 
        elif selected_tab == "Data Cleaning":
            st.header("Data Cleaning")
            data_cleaned = clean_data(data)
            st.write("Data after cleaning:")
            st.dataframe(data_cleaned.head())
 
            st.write("Basic Descriptive Statistics:")
            st.write(describe_data(data_cleaned))

            # Insight about data cleaning steps
            st.write("**Data Preparation Insight:**")
            st.markdown(
                """
                - Rows containing missing values were dropped to ensure a clean dataset.
                - This ensures that subsequent analysis is performed on a consistent and complete dataset.
                """
            )
 
        elif selected_tab == "Visualization":
            st.header("Visualization")
            data_cleaned = clean_data(data)
 
            # Correlation heatmap
            if st.checkbox("Show Correlation Heatmap"):
                corr = data_cleaned.corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                # Add insights
                st.write("**Correlation Insights:**")
                st.markdown(
                    """
                    - There is a strong positive correlation (0.62) between "Total" and "Results," indicating that total scores significantly influence the results.
                    - Most individual subjects show weak correlations with each other, suggesting relatively independent performance across subjects.
                    - A moderate correlation exists between all subjects and the "Total" column, which is expected since "Total" is the sum of all subjects.
                    """
                )

                        
            # Histograms
            if st.checkbox("Show Histograms"):
                for col in data_cleaned.select_dtypes(include=[np.number]).columns:
                    fig, ax = plt.subplots()
                    sns.histplot(data_cleaned[col], kde=True, ax=ax)
                    st.pyplot(fig)
 
        elif selected_tab == "K-means Clustering":
            st.header("K-means Clustering")
            data_cleaned = clean_data(data)
 
            st.write("Select features for clustering:")
            features_kmeans = st.multiselect("Features", options=data_cleaned.columns, default=data_cleaned.columns[:2])
 
            if len(features_kmeans) >= 2:
                X_kmeans = data_cleaned[features_kmeans].select_dtypes(include=[np.number])
                if not X_kmeans.empty:
                    k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)
 
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    data_cleaned['Cluster'] = kmeans.fit_predict(X_kmeans)
 
                    st.write("Cluster Centers:")
                    st.write(kmeans.cluster_centers_)
 
                    fig, ax = plt.subplots()
                    sns.scatterplot(
                        x=X_kmeans.iloc[:, 0], y=X_kmeans.iloc[:, 1], hue=data_cleaned['Cluster'], palette='viridis', ax=ax
                    )
                    plt.title("K-means Clustering")
                    plt.xlabel(features_kmeans[0])
                    plt.ylabel(features_kmeans[1])
                    st.pyplot(fig)
                else:
                    st.warning("Selected features do not contain any numeric data for clustering.")
 
        elif selected_tab == "Linear Regression":
            st.header("Linear Regression")
            data_cleaned = clean_data(data)
 
            st.write("Select target variable and predictors:")
            target = st.selectbox("Target variable", options=data_cleaned.columns)
            predictors = st.multiselect("Predictor variables", options=data_cleaned.columns, default=[col for col in data_cleaned.columns if col != target])
 
            if target and predictors:
                X = data_cleaned[predictors].select_dtypes(include=[np.number])
                y = data_cleaned[target]
 
                if not X.empty and not y.empty:
                    # Split data
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
                        # Train regression model
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
 
                        st.write("Regression Coefficients:")
                        coeff_df = pd.DataFrame({"Feature": predictors, "Coefficient": model.coef_})
                        st.write(coeff_df)
 
                        st.write("Model Performance:")
                        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
                        st.write(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")
 
                        # Visualization
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        plt.title("Actual vs Predicted")
                        plt.xlabel("Actual")
                        plt.ylabel("Predicted")
                        st.pyplot(fig)
                    except ValueError as e:
                        st.error(f"Error during train-test split or model training: {e}")
                else:
                    st.warning("Selected features or target variable do not contain sufficient data.")
 
        elif selected_tab == "Conclusions":
            st.header("Conclusions and Recommendations")
            st.write("Based on the clustering and regression analysis, here are the key takeaways:")
            
            st.subheader("Subject Independence")
            st.markdown('''- Correlation Heatmap reveals weak correlation between individual subjects (mostly below 0.1). This suggests that performance in one subject doesn't strongly predict performance in others.''')
            st.markdown('''- Students appear to have distinct strengths and weaknesses across different subjects.''')

            st.subheader("Performance Distribution")
            st.write("Mean scores across subjects are remarkably consistent, clustering around 50%:")
            st.markdown('''- Hindi: 51.6%''')
            st.markdown('''- English: 50.1%''')
            st.markdown('''- Science: 49.4%''')
            st.markdown('''- Maths: 49.6%''')
            st.markdown('''- History: 49.0%''')
            st.markdown('''- Geography: 50.0%''')

            st.subheader("Student Clustering")
            st.write("K-means clustering (k=3) identified distinct groups of students based on Hindi and English performance:")
            st.markdown('''- High performers in both subjects (yellow cluster)''')
            st.markdown('''- Strong Hindi but weaker English performance (purple cluster)''')
            st.markdown('''- Weaker performance in both subjects (teal cluster)''')
            st.write("This suggests potential language-based learning patterns.")
 
else:
    st.write("Please upload a dataset to proceed.")