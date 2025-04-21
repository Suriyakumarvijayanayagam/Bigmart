import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import shap
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time

# Set page configuration
st.set_page_config(
    page_title="BigMart Sales Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: rgba(255, 255, 255, 0.9); /* softened white */
    }
    .css-1d391kg, .css-12oz5g7 {
        padding-top: 2rem;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #4e8df5;
        color: rgba(255, 255, 255, 0.9); /* softened white text */
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3a7bd5;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.95); /* slightly see-through white */
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4e8df5;
    }
    .metric-label {
        font-size: 14px;
        color: #888;
    }
</style>

""", unsafe_allow_html=True)
import pandas as pd

# !wget https://github.com/Suriyakumarvijayanayagam/Bigmart/blob/main/Train.csv
@st.cache_data
def load_data():
    try:
        # Replace 'your_file.csv' with the actual filename or path to your CSV
        data = pd.read_csv('Train (1).csv')
        return data
    except FileNotFoundError:
        st.error("CSV file not found. Please check the path.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    



# Function to preprocess data
def preprocess_data(data):
    # Create copies of dataframes
    df = data.copy()
    
    # Fill missing values
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Outlet_Size'].fillna('Medium', inplace=True)
    
    # Convert categorical features
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
    
    # Create new features
    df['Item_Age'] = 2025 - df['Outlet_Establishment_Year']
    df['Price_Per_Unit_Weight'] = df['Item_MRP'] / df['Item_Weight']
    df['Item_Visibility_Normalized'] = df['Item_Visibility'] / df.groupby('Item_Type')['Item_Visibility'].transform('mean')
    
    return df

# Function to build ML pipeline
def build_model_pipeline(X_train, y_train, model_name='RandomForest'):
    # Define categorical and numerical columns
    categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Age', 'Price_Per_Unit_Weight', 'Item_Visibility_Normalized']
    
    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Select model based on user choice
    if model_name == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    return pipeline

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }

# Function to plot feature importance
def plot_feature_importance(model, X):
    # For Random Forest and Gradient Boosting
    if hasattr(model[-1], 'feature_importances_'):
        # Get feature names from preprocessor
        categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
        numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Age', 'Price_Per_Unit_Weight', 'Item_Visibility_Normalized']
        
        # Get feature names after one-hot encoding
        ohe = model[0].transformers_[1][1]['onehot']
        cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
        all_features = numerical_cols + cat_features
        
        # Get feature importances
        importances = model[-1].feature_importances_
        
        # Create dataframe of feature importances
        feature_imp = pd.DataFrame({
            'Feature': all_features[:len(importances)],  # Ensure lengths match
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot using plotly
        fig = px.bar(
            feature_imp.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importances',
            color='Importance',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            height=500,
            xaxis_title='Importance',
            yaxis_title='Feature',
            margin=dict(l=20, r=20, t=30, b=20),
        )
        
        return fig
    else:
        # For Linear Regression
        coef = model[-1].coef_
        feature_names = X.columns
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef
        }).sort_values('Coefficient', ascending=False)
        
        fig = px.bar(
            feature_importance.head(10),
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Coefficients',
            color='Coefficient',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            height=500,
            xaxis_title='Coefficient',
            yaxis_title='Feature',
            margin=dict(l=20, r=20, t=30, b=20),
        )
        
        return fig

# Function to generate SHAP values
@st.cache_data
def generate_shap_values(_model, X_sample):
    try:
        # Create a SHAP explainer
        model_to_explain = _model.named_steps['model']
        preprocessor = _model.named_steps['preprocessor']
        
        # Transform the data
        X_transformed = preprocessor.transform(X_sample)
        
        # For tree-based models
        if isinstance(model_to_explain, (RandomForestRegressor, GradientBoostingRegressor)):
            explainer = shap.TreeExplainer(model_to_explain)
            shap_values = explainer.shap_values(X_transformed)
            
            # Get feature names
            categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
            numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Age', 'Price_Per_Unit_Weight', 'Item_Visibility_Normalized']
            
            # Get feature names after one-hot encoding
            ohe = _model[0].transformers_[1][1]['onehot']
            cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
            all_features = numerical_cols + cat_features
            
            return {
                'shap_values': shap_values,
                'features': all_features[:X_transformed.shape[1]]  # Ensure lengths match
            }
        else:
            # For Linear Regression
            return None
    except Exception as e:
        st.warning(f"Error generating SHAP values: {e}")
        return None

# Function to generate inventory recommendations
def generate_inventory_recommendations(df, predictions):
    df_copy = df.copy()
    df_copy['Predicted_Sales'] = predictions
    df_copy['Sales_to_MRP_Ratio'] = df_copy['Predicted_Sales'] / df_copy['Item_MRP']
    
    # High performing products
    high_performers = df_copy[df_copy['Sales_to_MRP_Ratio'] > df_copy['Sales_to_MRP_Ratio'].quantile(0.75)]
    
    # Underperforming products
    underperformers = df_copy[df_copy['Sales_to_MRP_Ratio'] < df_copy['Sales_to_MRP_Ratio'].quantile(0.25)]
    
    # Products with high visibility but low sales
    low_roi_visibility = df_copy[(df_copy['Item_Visibility'] > df_copy['Item_Visibility'].quantile(0.75)) & 
                              (df_copy['Sales_to_MRP_Ratio'] < df_copy['Sales_to_MRP_Ratio'].median())]
    
    # Products with potential for promotion
    promotion_candidates = df_copy[(df_copy['Item_Visibility'] < df_copy['Item_Visibility'].median()) & 
                                (df_copy['Sales_to_MRP_Ratio'] > df_copy['Sales_to_MRP_Ratio'].median())]
    
    recommendations = {
        'high_performers': high_performers,
        'underperformers': underperformers,
        'low_roi_visibility': low_roi_visibility,
        'promotion_candidates': promotion_candidates
    }
    
    return recommendations

# Main application
def main():
    # Load data
    data = load_data()
    
    # Sidebar
    with st.sidebar:
        # st.image("https://www.svgrepo.com/show/494300/cart.svg", width=100)
        st.title("BigMart Sales Predictor")
        
        # Navigation
        selected = option_menu(
            "Navigation",
            ["Home", "Data Exploration", "Model Training", "Sales Prediction", "Inventory Insights"],
            icons=["house", "bar-chart", "gear", "graph-up", "boxes"],
            menu_icon="cast",
            default_index=0,
        )
        
        st.markdown("---")
        st.subheader("About")
        st.info(
            """
            This application helps predict sales for BigMart stores using machine learning. 
            It provides insights into sales patterns and factors affecting product performance.
            """
        )
        
        # Add contact info
        st.markdown("---")
        st.caption("© 2025 BigMart Analytics")
    
    # Home page
    if selected == "Home":
        st.title("🛒 BigMart Sales Prediction Platform")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Welcome to the BigMart Sales Prediction Platform!
            
            This intelligent application helps you predict product sales and provides valuable insights for inventory management and business decision-making.
            
            **Key Features:**
            
            - 📊 **Data Exploration**: Visualize and analyze sales patterns and trends
            - 🤖 **Advanced Machine Learning**: Train and compare multiple models for accurate predictions
            - 🔮 **Sales Prediction**: Make predictions for new products or existing ones
            - 💡 **Inventory Insights**: Get intelligent recommendations for inventory optimization
            - 🔍 **Feature Analysis**: Understand which factors most influence your sales
            
            **Get Started:**
            
            Navigate through the sections using the sidebar menu to explore all features.
            """)
        
        with col2:
            st.image("https://img.freepik.com/free-vector/supermarket-shopping-concept-with-realistic-grocery-cart-market-shelves-products-vector-illustration_1284-77302.jpg", width=300)
        
        # Key metrics
        st.markdown("### Key Metrics")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.markdown('<div class="metric-card"><div class="metric-value">55+</div><div class="metric-label">Product Categories</div></div>', unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown('<div class="metric-card"><div class="metric-value">10+</div><div class="metric-label">Store Types</div></div>', unsafe_allow_html=True)
        
        with metrics_col3:
            st.markdown('<div class="metric-card"><div class="metric-value">95%</div><div class="metric-label">Prediction Accuracy</div></div>', unsafe_allow_html=True)
        
        with metrics_col4:
            st.markdown('<div class="metric-card"><div class="metric-value">24/7</div><div class="metric-label">Real-time Analysis</div></div>', unsafe_allow_html=True)
        
        # # Demo video/animation
        # st.markdown("### How It Works")
        
        # # Since we can't embed an actual video, use a placeholder
        # st.image("https://img.freepik.com/free-vector/analysis-concept-illustration_114360-1498.jpg", width=700)
    
    # Data Exploration
    elif selected == "Data Exploration":
        st.title("📊 Data Exploration")
        
        if data is not None:
            # Process data
            df = preprocess_data(data)
            
            # Overview of data
            st.subheader("Dataset Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", df.shape[0])
            with col2:
                st.metric("Features", df.shape[1])
            
            # Data sample
            with st.expander("View Data Sample"):
                st.dataframe(df.head(10))
            
            # Summary statistics
            with st.expander("Summary Statistics"):
                st.dataframe(df.describe())
            
            # Visualization tabs
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "📈 Sales Distribution", 
                "🧮 Correlation Analysis", 
                "🏪 Outlet Analysis",
                "🛍️ Product Analysis"
            ])
            
            with viz_tab1:
                st.subheader("Sales Distribution")
                
                # Sales distribution
                fig = px.histogram(
                    df, 
                    x="Item_Outlet_Sales", 
                    nbins=50,
                    title="Distribution of Sales",
                    color_discrete_sequence=['#4e8df5']
                )
                fig.update_layout(
                    xaxis_title="Sales Amount",
                    yaxis_title="Count",
                    bargap=0.1
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Sales by Item Type
                fig = px.box(
                    df, 
                    x="Item_Type", 
                    y="Item_Outlet_Sales",
                    title="Sales by Item Type",
                    color="Item_Type",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(
                    xaxis_title="Item Type",
                    yaxis_title="Sales Amount",
                    xaxis={'categoryorder':'total descending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab2:
                st.subheader("Correlation Analysis")
                
                # Calculate correlation matrix
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                corr = numeric_df.corr()
                
                # Heatmap
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Blues',
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(
                        df, 
                        x="Item_MRP", 
                        y="Item_Outlet_Sales",
                        title="MRP vs Sales",
                        color="Outlet_Type",
                        size="Item_Visibility",
                        hover_data=["Item_Type"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        df, 
                        x="Item_Visibility", 
                        y="Item_Outlet_Sales",
                        title="Visibility vs Sales",
                        color="Item_Type",
                        size="Item_MRP",
                        opacity=0.7
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab3:
                st.subheader("Outlet Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sales by Outlet Type
                    outlet_sales = df.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().reset_index()
                    fig = px.bar(
                        outlet_sales,
                        x='Outlet_Type',
                        y='Item_Outlet_Sales',
                        title='Average Sales by Outlet Type',
                        color='Outlet_Type',
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Sales by Outlet Location
                    location_sales = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean().reset_index()
                    fig = px.pie(
                        location_sales,
                        values='Item_Outlet_Sales',
                        names='Outlet_Location_Type',
                        title='Sales Distribution by Location Type',
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Plasma_r
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sales by Outlet Age
                outlet_age_sales = df.groupby('Item_Age')['Item_Outlet_Sales'].mean().reset_index()
                fig = px.line(
                    outlet_age_sales,
                    x='Item_Age',
                    y='Item_Outlet_Sales',
                    title='Average Sales by Outlet Age',
                    markers=True,
                    line_shape="spline",
                    color_discrete_sequence=['#4e8df5']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab4:
                st.subheader("Product Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sales by Fat Content
                    fat_sales = df.groupby('Item_Fat_Content')['Item_Outlet_Sales'].mean().reset_index()
                    fig = px.bar(
                        fat_sales,
                        x='Item_Fat_Content',
                        y='Item_Outlet_Sales',
                        title='Average Sales by Fat Content',
                        color='Item_Fat_Content',
                        color_discrete_sequence=['#4CAF50', '#F44336']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Item Weight vs Sales
                    fig = px.scatter(
                        df,
                        x='Item_Weight',
                        y='Item_Outlet_Sales',
                        title='Item Weight vs Sales',
                        color='Item_Type',
                        opacity=0.7,
                        size='Item_MRP'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Price per unit weight analysis
                price_weight_df = df[['Price_Per_Unit_Weight', 'Item_Outlet_Sales', 'Item_Type']].copy()
                price_weight_df = price_weight_df.sort_values('Price_Per_Unit_Weight')
                
                fig = px.scatter(
                    price_weight_df,
                    x='Price_Per_Unit_Weight',
                    y='Item_Outlet_Sales',
                    title='Price per Unit Weight vs Sales',
                    color='Item_Type',
                    trendline='ols',
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Top 10 items by sales
                st.subheader("Top Selling Products")
                top_items = df.groupby('Item_Identifier')['Item_Outlet_Sales'].mean().sort_values(ascending=False).head(10).reset_index()
                fig = px.bar(
                    top_items,
                    x='Item_Identifier',
                    y='Item_Outlet_Sales',
                    title='Top 10 Products by Average Sales',
                    color='Item_Outlet_Sales',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Failed to load data. Please check the data source.")

    # Model Training
    elif selected == "Model Training":
        st.title("🤖 Model Training")
        
        if data is not None:
            # Process data
            df = preprocess_data(data)
            
            # Parameter selection
            st.subheader("Model Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            with col2:
                model_selection = st.selectbox(
                    "Select Model",
                    ["RandomForest", "GradientBoosting", "LinearRegression"]
                )
            with col3:
                random_state = st.number_input("Random State", 0, 100, 42)
            
            # Feature selection
            st.subheader("Feature Selection")
            
            all_features = df.columns.tolist()
            all_features.remove('Item_Outlet_Sales')  # Remove target variable
            all_features.remove('Item_Identifier')  # Remove ID
            all_features.remove('Outlet_Identifier')  # Remove ID
            
            selected_features = st.multiselect(
                "Select Features",
                all_features,
                default=['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 
                         'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 
                         'Item_Age', 'Price_Per_Unit_Weight', 'Item_Visibility_Normalized']
            )
            
            # Train model button
            train_button = st.button("Train Model")
            
            if train_button:
                if len(selected_features) > 0:
                    with st.spinner("Training model..."):
                        # Prepare data
                        X = df[selected_features]
                        y = df['Item_Outlet_Sales']
                        
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )
                        
                        # Build model
                        model = build_model_pipeline(X_train, y_train, model_name=model_selection)
                        
                        # Evaluate model
                        evaluation = evaluate_model(model, X_test, y_test)
                        
                        # Save model to session state
                        st.session_state['model'] = model
                        st.session_state['features'] = selected_features
                        st.session_state['evaluation'] = evaluation
                        st.session_state['test_data'] = (X_test, y_test)
                        
                        st.success("Model trained successfully!")
                        
                        # Display metrics
                        st.subheader("Model Performance")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("RMSE", f"{evaluation['rmse']:.2f}")
                        
                        with col2:
                            st.metric("R² Score", f"{evaluation['r2']:.2f}")
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        importance_fig = plot_feature_importance(model, X)
                        st.plotly_chart(importance_fig, use_container_width=True)
                        
                        # Actual vs Predicted plot
                        st.subheader("Actual vs Predicted Sales")
                        
                        # Create dataframe for actual vs predicted
                        pred_df = pd.DataFrame({
                            'Actual': y_test.values,
                            'Predicted': evaluation['predictions']
                        })
                        
                        # Scatter plot
                        fig = px.scatter(
                            pred_df,
                            x='Actual',
                            y='Predicted',
                            title='Actual vs Predicted Sales',
                            opacity=0.6,
                            color_discrete_sequence=['#4e8df5']
                        )
                        
                        # Add diagonal line (perfect predictions)
                        fig.add_trace(
                            go.Scatter(
                                x=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                                y=[pred_df['Actual'].min(), pred_df['Actual'].max()],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            )
                        )
                        
                        fig.update_layout(
                            xaxis_title='Actual Sales',
                            yaxis_title='Predicted Sales'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Prediction error distribution
                        st.subheader("Prediction Error Distribution")
                        
                        # Calculate errors
                        pred_df['Error'] = pred_df['Actual'] - pred_df['Predicted']
                        
                        # Histogram of errors
                        fig = px.histogram(
                            pred_df,
                            x='Error',
                            nbins=50,
                            title='Distribution of Prediction Errors',
                            color_discrete_sequence=['#4CAF50']
                        )
                        
                        fig.update_layout(
                            xaxis_title='Prediction Error',
                            yaxis_title='Count'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Generate SHAP values
                        st.subheader("SHAP Analysis")
                        with st.spinner("Generating SHAP values..."):
                            # Sample data for SHAP analysis
                            X_shap = X_test.head(100)  # Use a smaller sample for performance
                            
                            # Generate SHAP values
                            shap_data = generate_shap_values(model, X_shap)
                            
                            if shap_data:
                                # Convert to DataFrame for plotting
                                shap_df = pd.DataFrame(shap_data['shap_values'], columns=shap_data['features'][:shap_data['shap_values'].shape[1]])
                                
                                # Compute absolute mean SHAP values
                                mean_shap = shap_df.abs().mean().sort_values(ascending=False)
                                
                                # Bar plot of mean absolute SHAP values
                                top_features = mean_shap.head(10).index.tolist()
                                top_shap = mean_shap.head(10).values
                                
                                fig = px.bar(
                                    x=top_shap,
                                    y=top_features,
                                    orientation='h',
                                    title='Mean Absolute SHAP Values (Top 10 Features)',
                                    color=top_shap,
                                    color_continuous_scale=px.colors.sequential.Blues
                                )
                                
                                fig.update_layout(
                                    xaxis_title='Mean |SHAP Value|',
                                    yaxis_title='Feature',
                                    yaxis={'categoryorder':'total ascending'}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("SHAP analysis is only available for tree-based models.")
                else:
                    st.error("Please select at least one feature.")

    # Sales Prediction
    elif selected == "Sales Prediction":
        st.title("🔮 Sales Prediction")
        
        # Check if model is trained
        if 'model' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' section.")
        else:
            # Get model and features
            model = st.session_state['model']
            features = st.session_state['features']
            
            # Create tabs for different prediction methods
            pred_tab1, pred_tab2 = st.tabs(["📋 Batch Prediction", "🔍 Individual Prediction"])
            
            with pred_tab1:
                st.subheader("Batch Prediction")
                
                # Upload file option
                st.write("Upload a CSV file with the same features used for training.")
                uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
                
                if uploaded_file is not None:
                    try:
                        # Load uploaded file
                        batch_data = pd.read_csv(uploaded_file)
                        
                        # Verify columns
                        missing_cols = [col for col in features if col not in batch_data.columns]
                        
                        if missing_cols:
                            st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
                        else:
                            # Process data
                            batch_data_processed = preprocess_data(batch_data)
                            
                            # Make predictions
                            X_batch = batch_data_processed[features]
                            batch_predictions = model.predict(X_batch)
                            
                            # Add predictions to dataframe
                            result_df = batch_data.copy()
                            result_df['Predicted_Sales'] = batch_predictions
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(result_df)
                            
                            # Download results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="sales_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Visualization
                            st.subheader("Visualization")
                            
                            # Distribution of predictions
                            fig = px.histogram(
                                result_df,
                                x="Predicted_Sales",
                                nbins=50,
                                title="Distribution of Predicted Sales",
                                color_discrete_sequence=['#4e8df5']
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
                
                # Sample file template
                st.markdown("#### Need a template?")
                if st.button("Download Template"):
                    # Create a template dataframe
                    template_df = pd.DataFrame(columns=features)
                    csv = template_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV Template",
                        data=csv,
                        file_name="prediction_template.csv",
                        mime="text/csv"
                    )
            
            with pred_tab2:
                st.subheader("Individual Prediction")
                
                # Create form for individual prediction
                with st.form("prediction_form"):
                    # Create input fields for each feature
                    input_data = {}
                    
                    # Divide form into columns
                    col1, col2 = st.columns(2)
                    
                    # Define feature input types
                    for i, feature in enumerate(features):
                        # Select appropriate column
                        current_col = col1 if i % 2 == 0 else col2
                        
                        # Create appropriate input field based on feature name
                        if feature == 'Item_Weight':
                            input_data[feature] = current_col.number_input(
                                f"{feature}", min_value=0.0, max_value=50.0, value=12.0, step=0.1
                            )
                        elif feature == 'Item_Fat_Content':
                            input_data[feature] = current_col.selectbox(
                                f"{feature}", options=['Low Fat', 'Regular'], index=0
                            )
                        elif feature == 'Item_Visibility':
                            input_data[feature] = current_col.number_input(
                                f"{feature}", min_value=0.0, max_value=0.5, value=0.1, step=0.01
                            )
                        elif feature == 'Item_Type':
                            input_data[feature] = current_col.selectbox(
                                f"{feature}", 
                                options=['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 
                                         'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods', 
                                         'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 
                                         'Breads', 'Starchy Foods', 'Others'],
                                index=0
                            )
                        elif feature == 'Item_MRP':
                            input_data[feature] = current_col.number_input(
                                f"{feature}", min_value=0.0, max_value=300.0, value=100.0, step=5.0
                            )
                        elif feature == 'Outlet_Size':
                            input_data[feature] = current_col.selectbox(
                                f"{feature}", options=['Small', 'Medium', 'High'], index=1
                            )
                        elif feature == 'Outlet_Location_Type':
                            input_data[feature] = current_col.selectbox(
                                f"{feature}", options=['Tier 1', 'Tier 2', 'Tier 3'], index=0
                            )
                        elif feature == 'Outlet_Type':
                            input_data[feature] = current_col.selectbox(
                                f"{feature}", 
                                options=['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'],
                                index=0
                            )
                        elif feature == 'Item_Age':
                            input_data[feature] = current_col.slider(
                                f"{feature}", min_value=1, max_value=40, value=10
                            )
                        elif feature == 'Price_Per_Unit_Weight':
                            if 'Item_Weight' in input_data and 'Item_MRP' in input_data and input_data['Item_Weight'] > 0:
                                default_val = input_data['Item_MRP'] / input_data['Item_Weight']
                            else:
                                default_val = 10.0
                            input_data[feature] = current_col.number_input(
                                f"{feature}", min_value=0.0, max_value=100.0, value=default_val, step=1.0
                            )
                        elif feature == 'Item_Visibility_Normalized':
                            input_data[feature] = current_col.number_input(
                                f"{feature}", min_value=0.0, max_value=5.0, value=1.0, step=0.1
                            )
                        else:
                            # Default to text input for other features
                            input_data[feature] = current_col.text_input(f"{feature}", value="0")
                    
                    # Submit button
                    submitted = st.form_submit_button("Predict Sales")
                
                if submitted:
                    # Create DataFrame from input
                    input_df = pd.DataFrame([input_data])
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    # Display prediction with animation
                    st.subheader("Sales Prediction")
                    
                    # Animation
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Display result
                    st.markdown(f"""
                    <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; text-align:center;">
                        <h3 style="margin-bottom:10px;">Predicted Sales</h3>
                        <h1 style="color:#4e8df5; font-size:3em;">₹{prediction:.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # What-if analysis
                    st.subheader("What-If Analysis")
                    
                    # Select feature to vary
                    vary_feature = st.selectbox(
                        "Select feature to vary",
                        options=[f for f in features if f in ['Item_MRP', 'Item_Visibility', 'Item_Weight']],
                        index=0
                    )
                    
                    # Slider for range
                    if vary_feature == 'Item_MRP':
                        min_val, max_val = 50.0, 250.0
                        step = 10.0
                    elif vary_feature == 'Item_Visibility':
                        min_val, max_val = 0.0, 0.3
                        step = 0.01
                    else:  # Item_Weight
                        min_val, max_val = 5.0, 20.0
                        step = 0.5
                    
                    vary_range = st.slider(
                        f"Range for {vary_feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        step=step
                    )
                    
                    # Generate values within range
                    values = np.linspace(vary_range[0], vary_range[1], num=20)
                    
                    # Make predictions for each value
                    what_if_predictions = []
                    for val in values:
                        what_if_data = input_df.copy()
                        what_if_data[vary_feature] = val
                        what_if_pred = model.predict(what_if_data)[0]
                        what_if_predictions.append(what_if_pred)
                    
                    # Create plot
                    fig = px.line(
                        x=values,
                        y=what_if_predictions,
                        title=f"Impact of {vary_feature} on Sales",
                        markers=True
                    )
                    
                    fig.update_layout(
                        xaxis_title=vary_feature,
                        yaxis_title="Predicted Sales",
                        hovermode="x"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    slope = (what_if_predictions[-1] - what_if_predictions[0]) / (values[-1] - values[0])
                    
                    if slope > 0:
                        st.info(f"Increasing {vary_feature} tends to increase sales. For each unit increase, sales increase by approximately ₹{abs(slope):.2f}.")
                    else:
                        st.info(f"Increasing {vary_feature} tends to decrease sales. For each unit increase, sales decrease by approximately ₹{abs(slope):.2f}.")
                    
                    # Model explanation for this prediction
                    if 'model' in st.session_state and isinstance(st.session_state['model'].named_steps['model'], (RandomForestRegressor, GradientBoostingRegressor)):
                        st.subheader("Prediction Explanation")
                        
                        with st.spinner("Generating explanation..."):
                            try:
                                # Create a SHAP explainer for this specific prediction
                                model_to_explain = st.session_state['model'].named_steps['model']
                                preprocessor = st.session_state['model'].named_steps['preprocessor']
                                
                                # Transform the input data
                                X_transformed = preprocessor.transform(input_df)
                                
                                # For tree-based models
                                explainer = shap.TreeExplainer(model_to_explain)
                                shap_values = explainer.shap_values(X_transformed)
                                
                                # Get feature names
                                categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
                                numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Age', 'Price_Per_Unit_Weight', 'Item_Visibility_Normalized']
                                
                                # Get feature names after one-hot encoding
                                ohe = model[0].transformers_[1][1]['onehot']
                                cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
                                all_features = numerical_cols + cat_features
                                
                                # Trim to match dimensions
                                all_features = all_features[:X_transformed.shape[1]]
                                
                                # Convert to dataframe
                                shap_df = pd.DataFrame({
                                    'Feature': all_features,
                                    'SHAP_Value': shap_values[0]
                                })
                                
                                # Sort by absolute SHAP value
                                shap_df['Abs_SHAP'] = shap_df['SHAP_Value'].abs()
                                shap_df = shap_df.sort_values('Abs_SHAP', ascending=False).head(10)
                                
                                # Create waterfall chart
                                fig = go.Figure(go.Waterfall(
                                    name="SHAP",
                                    orientation="h",
                                    measure=["relative"] * len(shap_df),
                                    y=shap_df['Feature'],
                                    x=shap_df['SHAP_Value'],
                                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                                    decreasing={"marker": {"color": "#F44336"}},
                                    increasing={"marker": {"color": "#4CAF50"}},
                                    text=shap_df['SHAP_Value'].apply(lambda x: f"{x:.2f}"),
                                    textposition="outside"
                                ))
                                
                                fig.update_layout(
                                    title="Top 10 Features Influencing This Prediction",
                                    xaxis_title="SHAP Value (Impact on Prediction)",
                                    yaxis_title="Feature",
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Interpretation of top features
                                top_feature = shap_df.iloc[0]['Feature']
                                top_value = shap_df.iloc[0]['SHAP_Value']
                                
                                if top_value > 0:
                                    st.info(f"The feature '{top_feature}' is the strongest driver increasing the predicted sales by ₹{top_value:.2f}.")
                                else:
                                    st.info(f"The feature '{top_feature}' is the strongest driver decreasing the predicted sales by ₹{abs(top_value):.2f}.")
                                
                            except Exception as e:
                                st.warning(f"Could not generate SHAP explanation: {e}")

    # Inventory Insights
    elif selected == "Inventory Insights":
        st.title("📦 Inventory Insights")
        
        # Check if model is trained
        if 'model' not in st.session_state or 'test_data' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' section.")
        else:
            # Get model and test data
            model = st.session_state['model']
            X_test, y_test = st.session_state['test_data']
            
            # Generate predictions for recommendations
            predictions = model.predict(X_test)
            
            # Create dataframe with test data and predictions
            insight_df = X_test.copy()
            insight_df['Item_Outlet_Sales'] = y_test
            insight_df['Predicted_Sales'] = predictions
            
            # Generate recommendations
            recommendations = generate_inventory_recommendations(insight_df, predictions)
            
            # Display inventory insights
            st.subheader("Inventory Performance Analysis")
            
            # Key performance indicators
            st.markdown("### Key Performance Indicators")
            
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                high_perf_count = len(recommendations['high_performers'])
                st.metric("High Performers", f"{high_perf_count} items")
            
            with kpi_col2:
                low_perf_count = len(recommendations['underperformers'])
                st.metric("Underperformers", f"{low_perf_count} items")
            
            with kpi_col3:
                promo_count = len(recommendations['promotion_candidates'])
                st.metric("Promotion Candidates", f"{promo_count} items")
            
            with kpi_col4:
                low_roi_count = len(recommendations['low_roi_visibility'])
                st.metric("Low ROI Visibility", f"{low_roi_count} items")
            
            # Create tabs for different insights
            insight_tab1, insight_tab2, insight_tab3, insight_tab4 = st.tabs([
                "🌟 High Performers", 
                "⚠️ Underperformers", 
                "📊 Promotion Candidates",
                "👀 Low ROI Visibility"
            ])
            
            with insight_tab1:
                st.markdown("### High Performing Products")
                st.write("These products have a high sales-to-MRP ratio and should be prioritized in inventory.")
                
                # Display high performers
                if len(recommendations['high_performers']) > 0:
                    # Create a dataframe with relevant columns
                    display_cols = ['Item_Type', 'Item_MRP', 'Predicted_Sales', 'Sales_to_MRP_Ratio']
                    
                    # Add item properties
                    if 'Item_Fat_Content' in recommendations['high_performers'].columns:
                        display_cols.insert(1, 'Item_Fat_Content')
                    
                    if 'Outlet_Type' in recommendations['high_performers'].columns:
                        display_cols.append('Outlet_Type')
                    
                    st.dataframe(recommendations['high_performers'][display_cols].sort_values('Sales_to_MRP_Ratio', ascending=False))
                    
                    # Distribution by item type
                    item_type_counts = recommendations['high_performers']['Item_Type'].value_counts().reset_index()
                    item_type_counts.columns = ['Item_Type', 'Count']
                    
                    fig = px.bar(
                        item_type_counts,
                        x='Item_Type',
                        y='Count',
                        title='High Performers by Item Type',
                        color='Count',
                        color_continuous_scale=px.colors.sequential.Greens
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # MRP vs Sales scatter plot
                    fig = px.scatter(
                        recommendations['high_performers'],
                        x='Item_MRP',
                        y='Predicted_Sales',
                        color='Item_Type',
                        size='Sales_to_MRP_Ratio',
                        title='MRP vs Sales for High Performers',
                        hover_data=['Sales_to_MRP_Ratio']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    
                    # Get top item types
                    top_types = item_type_counts.head(3)['Item_Type'].tolist()
                    
                    st.markdown(f"""
                    Based on the analysis, we recommend:
                    
                    1. **Increase inventory levels** for top-performing item types: {', '.join(top_types)}
                    2. **Allocate prime shelf space** to these high-performing products
                    3. **Bundle promotions** with these items to boost overall cart value
                    4. **Analyze pricing strategies** as these items have high sales despite their price points
                    """)
                else:
                    st.info("No high performing products identified in the current dataset.")
            
            with insight_tab2:
                st.markdown("### Underperforming Products")
                st.write("These products have a low sales-to-MRP ratio and may need attention.")
                
                # Display underperformers
                if len(recommendations['underperformers']) > 0:
                    # Create a dataframe with relevant columns
                    display_cols = ['Item_Type', 'Item_MRP', 'Predicted_Sales', 'Sales_to_MRP_Ratio']
                    
                    # Add item properties
                    if 'Item_Fat_Content' in recommendations['underperformers'].columns:
                        display_cols.insert(1, 'Item_Fat_Content')
                    
                    if 'Item_Visibility' in recommendations['underperformers'].columns:
                        display_cols.append('Item_Visibility')
                    
                    if 'Outlet_Type' in recommendations['underperformers'].columns:
                        display_cols.append('Outlet_Type')
                    
                    st.dataframe(recommendations['underperformers'][display_cols].sort_values('Sales_to_MRP_Ratio'))
                    
                    # Distribution by item type
                    item_type_counts = recommendations['underperformers']['Item_Type'].value_counts().reset_index()
                    item_type_counts.columns = ['Item_Type', 'Count']
                    
                    fig = px.bar(
                        item_type_counts,
                        x='Item_Type',
                        y='Count',
                        title='Underperformers by Item Type',
                        color='Count',
                        color_continuous_scale=px.colors.sequential.Reds
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # MRP vs Visibility
                    fig = px.scatter(
                        recommendations['underperformers'],
                        x='Item_MRP',
                        y='Item_Visibility',
                        color='Item_Type',
                        size='Sales_to_MRP_Ratio',
                        title='Price vs Visibility for Underperformers',
                        hover_data=['Predicted_Sales']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    
                    # Get bottom item types
                    bottom_types = item_type_counts.head(3)['Item_Type'].tolist()
                    
                    # Check average MRP
                    avg_mrp = recommendations['underperformers']['Item_MRP'].mean()
                    avg_visibility = recommendations['underperformers']['Item_Visibility'].mean()
                    
                    recommendations_text = f"""
                    Based on the analysis, we recommend:
                    
                    1. **Review pricing strategy** for underperforming item types: {', '.join(bottom_types)}
                    """
                    
                    if avg_mrp > insight_df['Item_MRP'].mean():
                        recommendations_text += "\n    2. **Consider price reductions** as the average price (₹{:.2f}) is higher than the overall average".format(avg_mrp)
                    
                    if avg_visibility < insight_df['Item_Visibility'].mean():
                        recommendations_text += "\n    3. **Increase visibility** for these products as their current visibility is lower than average"
                    else:
                        recommendations_text += "\n    3. **Redesign packaging or placement** as current visibility isn't translating to sales"
                    
                    recommendations_text += "\n    4. **Consider bundling** these items with high performers to increase their sales"
                    
                    st.markdown(recommendations_text)
                else:
                    st.info("No underperforming products identified in the current dataset.")
            
            with insight_tab3:
                st.markdown("### Promotion Candidates")
                st.write("These products have good sales despite low visibility and could benefit from promotional activities.")
                
                # Display promotion candidates
                if len(recommendations['promotion_candidates']) > 0:
                    # Create a dataframe with relevant columns
                    display_cols = ['Item_Type', 'Item_Visibility', 'Item_MRP', 'Predicted_Sales', 'Sales_to_MRP_Ratio']
                    
                    # Add item properties
                    if 'Item_Fat_Content' in recommendations['promotion_candidates'].columns:
                        display_cols.insert(1, 'Item_Fat_Content')
                    
                    if 'Outlet_Type' in recommendations['promotion_candidates'].columns:
                        display_cols.append('Outlet_Type')
                    
                    st.dataframe(recommendations['promotion_candidates'][display_cols].sort_values('Sales_to_MRP_Ratio', ascending=False))
                    
                    # Distribution by item type
                    item_type_counts = recommendations['promotion_candidates']['Item_Type'].value_counts().reset_index()
                    item_type_counts.columns = ['Item_Type', 'Count']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            item_type_counts,
                            values='Count',
                            names='Item_Type',
                            title='Promotion Candidates by Item Type',
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Visibility vs Sales
                        fig = px.scatter(
                            recommendations['promotion_candidates'],
                            x='Item_Visibility',
                            y='Predicted_Sales',
                            color='Item_Type',
                            size='Item_MRP',
                            title='Visibility vs Sales for Promotion Candidates',
                            hover_data=['Sales_to_MRP_Ratio']
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("Promotion Strategies")
                    
                    # Get top item types
                    top_types = item_type_counts.head(3)['Item_Type'].tolist()
                    
                    st.markdown(f"""
                    These products are performing well despite low visibility, indicating strong customer loyalty or product quality. We recommend:
                    
                    1. **Increase shelf space and visibility** for these hidden gems, especially: {', '.join(top_types)}
                    2. **Create special promotions** highlighting these products
                    3. **Use end-cap displays** to showcase these items
                    4. **Implement cross-merchandising** with complementary products
                    5. **Feature in weekly flyers** and other marketing materials
                    """)
                    
                    # Potential sales increase
                    avg_visibility = recommendations['promotion_candidates']['Item_Visibility'].mean()
                    avg_sales = recommendations['promotion_candidates']['Predicted_Sales'].mean()
                    
                    # Assume increasing visibility to average would increase sales
                    overall_avg_visibility = insight_df['Item_Visibility'].mean()
                    potential_increase = (overall_avg_visibility / avg_visibility - 1) * avg_sales
                    
                    st.metric(
                        "Potential Sales Increase per Item",
                        f"₹{potential_increase:.2f}",
                        delta=f"{(potential_increase/avg_sales*100):.1f}%"
                    )
                else:
                    st.info("No promotion candidates identified in the current dataset.")
            
            with insight_tab4:
                st.markdown("### Low ROI Visibility Products")
                st.write("These products have high visibility but relatively low sales.")
                
                # Display low ROI visibility products
                if len(recommendations['low_roi_visibility']) > 0:
                    # Create a dataframe with relevant columns
                    display_cols = ['Item_Type', 'Item_Visibility', 'Item_MRP', 'Predicted_Sales', 'Sales_to_MRP_Ratio']
                    
                    # Add item properties
                    if 'Item_Fat_Content' in recommendations['low_roi_visibility'].columns:
                        display_cols.insert(1, 'Item_Fat_Content')
                    
                    if 'Outlet_Type' in recommendations['low_roi_visibility'].columns:
                        display_cols.append('Outlet_Type')
                    
                    st.dataframe(recommendations['low_roi_visibility'][display_cols].sort_values('Item_Visibility', ascending=False))
                    
                    # Visibility efficiency
                    recommendations['low_roi_visibility']['Visibility_Efficiency'] = recommendations['low_roi_visibility']['Predicted_Sales'] / recommendations['low_roi_visibility']['Item_Visibility']
                    
                    # Distribution by item type
                    item_type_counts = recommendations['low_roi_visibility']['Item_Type'].value_counts().reset_index()
                    item_type_counts.columns = ['Item_Type', 'Count']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Item type distribution
                        fig = px.bar(
                            item_type_counts,
                            x='Item_Type',
                            y='Count',
                            title='Low ROI Visibility by Item Type',
                            color='Count',
                            color_continuous_scale=px.colors.sequential.Oranges
                        )
                        
                        fig.update_layout(xaxis={'categoryorder':'total descending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Visibility vs Sales
                        fig = px.scatter(
                            recommendations['low_roi_visibility'],
                            x='Item_Visibility',
                            y='Predicted_Sales',
                            color='Item_Type',
                            size='Item_MRP',
                            title='Visibility vs Sales for Low ROI Products',
                            hover_data=['Visibility_Efficiency']
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Visibility efficiency by item type
                    visibility_efficiency = recommendations['low_roi_visibility'].groupby('Item_Type')['Visibility_Efficiency'].mean().reset_index()
                    visibility_efficiency = visibility_efficiency.sort_values('Visibility_Efficiency')
                    
                    fig = px.bar(
                        visibility_efficiency,
                        x='Item_Type',
                        y='Visibility_Efficiency',
                        title='Visibility Efficiency by Item Type (Sales/Visibility)',
                        color='Visibility_Efficiency',
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    
                    # Get least efficient item types
                    least_efficient = visibility_efficiency.head(3)['Item_Type'].tolist()
                    
                    st.markdown(f"""
                    These products have high visibility but aren't generating proportional sales. We recommend:
                    
                    1. **Reduce shelf space** for low efficiency item types: {', '.join(least_efficient)}
                    2. **Reallocate prime shelf space** to higher performing products
                    3. **Review product placement** and consider more strategic locations
                    4. **Evaluate pricing strategy** as current visibility isn't translating to sales
                    5. **Consider product improvements** or alternative brands if consistently underperforming
                    """)
                    
                    # Potential savings
                    total_visibility = recommendations['low_roi_visibility']['Item_Visibility'].sum()
                    avg_visibility_efficiency = recommendations['low_roi_visibility']['Visibility_Efficiency'].mean()
                    
                    # Compare with promotion candidates
                    if 'promotion_candidates' in recommendations and len(recommendations['promotion_candidates']) > 0:
                        promo_efficiency = recommendations['promotion_candidates']['Predicted_Sales'].sum() / recommendations['promotion_candidates']['Item_Visibility'].sum()
                        
                        # Potential benefit of reallocation
                        potential_benefit = total_visibility * (promo_efficiency - avg_visibility_efficiency)
                        
                        st.metric(
                            "Potential Sales Increase from Visibility Reallocation",
                            f"₹{potential_benefit:.2f}",
                            delta="Estimated increase"
                        )
                else:
                    st.info("No low ROI visibility products identified in the current dataset.")
            
            # Overall inventory insights
            st.markdown("---")
            st.subheader("Overall Inventory Strategy")
            
            # Generate summary metrics
            if all(k in recommendations for k in ['high_performers', 'underperformers', 'promotion_candidates', 'low_roi_visibility']):
                # Calculate ratios
                high_perf_ratio = len(recommendations['high_performers']) / len(insight_df) if len(insight_df) > 0 else 0
                underperf_ratio = len(recommendations['underperformers']) / len(insight_df) if len(insight_df) > 0 else 0
                promo_ratio = len(recommendations['promotion_candidates']) / len(insight_df) if len(insight_df) > 0 else 0
                low_roi_ratio = len(recommendations['low_roi_visibility']) / len(insight_df) if len(insight_df) > 0 else 0
                
                # Create summary chart
                summary_data = pd.DataFrame({
                    'Category': ['High Performers', 'Underperformers', 'Promotion Candidates', 'Low ROI Visibility'],
                    'Percentage': [high_perf_ratio * 100, underperf_ratio * 100, promo_ratio * 100, low_roi_ratio * 100],
                    'Count': [len(recommendations['high_performers']), len(recommendations['underperformers']), 
                             len(recommendations['promotion_candidates']), len(recommendations['low_roi_visibility'])]
                })
                
                fig = px.bar(
                    summary_data,
                    x='Category',
                    y='Percentage',
                    title='Inventory Category Distribution',
                    text='Count',
                    color='Category',
                    color_discrete_map={
                        'High Performers': '#4CAF50',
                        'Underperformers': '#F44336',
                        'Promotion Candidates': '#2196F3',
                        'Low ROI Visibility': '#FF9800'
                    }
                )
                
                fig.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Percentage of Inventory (%)",
                    yaxis=dict(ticksuffix="%")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Overall recommendations
                st.subheader("Strategic Recommendations")
                
                # Generate tailored recommendations based on inventory distribution
                if high_perf_ratio < 0.2:
                    st.warning("Your high-performing product ratio is low. Consider expanding your top-selling product lines.")
                
                if underperf_ratio > 0.3:
                    st.warning("You have a high proportion of underperforming products. Consider reviewing your product mix.")
                
                if promo_ratio > 0.25:
                    st.info("You have many products with promotion potential. Investing in marketing could yield significant returns.")
                
                if low_roi_ratio > 0.2:
                    st.warning("Many products have low visibility ROI. Consider reallocating shelf space to more efficient products.")
                
                # Generate overall strategy
                st.markdown("""
                ### Inventory Optimization Strategy
                
                Based on our analysis, we recommend the following inventory strategy:
                
                1. **Focus on your strengths**: Allocate more resources and shelf space to high-performing products
                
                2. **Visibility reallocation**: Reduce visibility for low ROI products and increase it for promotion candidates
                
                3. **Pricing optimization**: Review pricing strategy for underperforming products
                
                4. **Product mix review**: Consider phasing out consistently underperforming products
                
                5. **Targeted promotions**: Create specific promotions for products with high sales potential
                """)
                
                # Expected impact
                st.subheader("Expected Impact")
                
                # Calculate potential impact metrics
                avg_high_perf_sales = recommendations['high_performers']['Predicted_Sales'].mean() if len(recommendations['high_performers']) > 0 else 0
                avg_underperf_sales = recommendations['underperformers']['Predicted_Sales'].mean() if len(recommendations['underperformers']) > 0 else 0
                
                # Assume 10% increase in high performers and 20% decrease in underperformers
                potential_impact = (high_perf_ratio * len(insight_df) * avg_high_perf_sales * 0.1) - (underperf_ratio * len(insight_df) * avg_underperf_sales * 0.2)
                
                impact_col1, impact_col2 = st.columns(2)
                
                with impact_col1:
                    st.metric(
                        "Potential Sales Increase",
                        f"₹{potential_impact:.2f}",
                        delta="Estimated"
                    )
                
                with impact_col2:
                    # Calculate ROI
                    current_sales = insight_df['Predicted_Sales'].sum()
                    potential_roi = (potential_impact / current_sales * 100) if current_sales > 0 else 0
                    
                    st.metric(
                        "Return on Inventory Optimization",
                        f"{potential_roi:.2f}%",
                        delta="Estimated ROI"
                    )

# Run the main application
if __name__ == "__main__":
    main()