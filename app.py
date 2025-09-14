import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="wide")

# Title and description
st.title("🏡 House Price Predictor")
st.markdown("**Predict house prices based on area, bedrooms, and bathrooms using machine learning**")

# Enhanced dataset with more realistic data
@st.cache_data
def load_data():
    np.random.seed(42)  # For reproducible results
    n_samples = 100
    
    # Generate synthetic but realistic data
    area = np.random.normal(2000, 500, n_samples)
    area = np.clip(area, 800, 4000)  # Realistic range
    
    bedrooms = np.random.choice([2, 3, 4, 5, 6], n_samples, p=[0.15, 0.35, 0.3, 0.15, 0.05])
    bathrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.4, 0.3, 0.15, 0.05])
    
    # Price based on realistic factors with some noise
    price = (area * 150 + bedrooms * 15000 + bathrooms * 10000 + 
             np.random.normal(0, 20000, n_samples))
    price = np.clip(price, 150000, 800000)  # Realistic price range
    
    return pd.DataFrame({
        "area": area.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "price": price.astype(int)
    })

df = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a section:", ["Prediction", "Data Analysis", "Model Performance"])

if page == "Prediction":
    # Main prediction interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 Enter House Details")
        area = st.number_input("Area (sq ft)", min_value=500, max_value=5000, value=1800, step=50)
        bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
        
        # Features and Target
        X = df[["area", "bedrooms", "bathrooms"]]
        y = df["price"]
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Prediction
        if st.button("🎯 Predict Price", type="primary"):
            prediction = model.predict([[area, bedrooms, bathrooms]])
            predicted_price = prediction[0]
            
            # Display prediction with confidence interval
            st.success(f"### Estimated Price: ${predicted_price:,.0f}")
            
            # Show feature contributions
            coefficients = model.coef_
            intercept = model.intercept_
            
            area_contribution = coefficients[0] * area
            bedrooms_contribution = coefficients[1] * bedrooms
            bathrooms_contribution = coefficients[2] * bathrooms
            
            st.write("#### Price Breakdown:")
            st.write(f"• Base price: ${intercept:,.0f}")
            st.write(f"• Area contribution ({area} sq ft): ${area_contribution:,.0f}")
            st.write(f"• Bedrooms contribution ({bedrooms}): ${bedrooms_contribution:,.0f}")
            st.write(f"• Bathrooms contribution ({bathrooms}): ${bathrooms_contribution:,.0f}")
    
    with col2:
        st.subheader("📊 Price Comparison")
        
        # Create comparison with similar houses
        similar_houses = df[
            (df['bedrooms'] == bedrooms) & 
            (df['bathrooms'] == bathrooms) &
            (abs(df['area'] - area) < 300)
        ].head(5)
        
        if len(similar_houses) > 0:
            st.write("Similar houses in our dataset:")
            comparison_df = similar_houses[['area', 'bedrooms', 'bathrooms', 'price']].copy()
            
            # Add the predicted house
            new_row = pd.DataFrame({
                'area': [area],
                'bedrooms': [bedrooms], 
                'bathrooms': [bathrooms],
                'price': [int(model.predict([[area, bedrooms, bathrooms]])[0])]
            })
            new_row.index = ['Predicted']
            
            comparison_df = pd.concat([comparison_df, new_row])
            comparison_df['Type'] = ['Market Data'] * len(similar_houses) + ['Prediction']
            
            # Create bar chart
            fig = px.bar(comparison_df.reset_index(), 
                        x='index', y='price',
                        color='Type',
                        title="Price Comparison",
                        labels={'index': 'House', 'price': 'Price ($)'})
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No similar houses found in the dataset for comparison.")

elif page == "Data Analysis":
    st.subheader("📈 Dataset Analysis")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Houses", len(df))
    with col2:
        st.metric("Avg Price", f"${df['price'].mean():,.0f}")
    with col3:
        st.metric("Avg Area", f"{df['area'].mean():.0f} sq ft")
    with col4:
        st.metric("Price Range", f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    
    # Display raw data
    st.subheader("🗂️ Training Dataset")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Area scatter plot
        fig1 = px.scatter(df, x='area', y='price', 
                         size='bedrooms', color='bathrooms',
                         title="Price vs Area (size=bedrooms, color=bathrooms)",
                         labels={'area': 'Area (sq ft)', 'price': 'Price ($)'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Price distribution
        fig2 = px.histogram(df, x='price', nbins=20,
                           title="Price Distribution",
                           labels={'price': 'Price ($)', 'count': 'Number of Houses'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Correlation matrix
    st.subheader("🔗 Feature Correlations")
    corr_matrix = df[['area', 'bedrooms', 'bathrooms', 'price']].corr()
    fig3 = px.imshow(corr_matrix, 
                     labels=dict(color="Correlation"),
                     title="Correlation Matrix")
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Model Performance":
    st.subheader("🎯 Model Performance Metrics")
    
    # Prepare data and train model
    X = df[["area", "bedrooms", "bathrooms"]]
    y = df["price"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score (Train)", f"{r2_score(y_train, y_pred_train):.3f}")
    with col2:
        st.metric("R² Score (Test)", f"{r2_score(y_test, y_pred_test):.3f}")
    with col3:
        st.metric("Mean Absolute Error", f"${mean_absolute_error(y_test, y_pred_test):,.0f}")
    
    # Model coefficients
    st.subheader("📋 Model Coefficients")
    coeff_df = pd.DataFrame({
        'Feature': ['Area (per sq ft)', 'Bedrooms', 'Bathrooms'],
        'Coefficient': model.coef_,
        'Impact': [f"${coeff:+.0f}" for coeff in model.coef_]
    })
    st.dataframe(coeff_df, use_container_width=True)
    
    st.write(f"**Base Price (Intercept):** ${model.intercept_:,.0f}")
    
    # Actual vs Predicted plot
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=y_test, y=y_pred_test, mode='markers',
                             name='Test Predictions'))
    fig4.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                             y=[y_test.min(), y_test.max()],
                             mode='lines', name='Perfect Prediction',
                             line=dict(dash='dash')))
    fig4.update_layout(title="Actual vs Predicted Prices",
                      xaxis_title="Actual Price ($)",
                      yaxis_title="Predicted Price ($)")
    st.plotly_chart(fig4, use_container_width=True)
    
    # Residuals plot
    residuals = y_test - y_pred_test
    fig5 = px.scatter(x=y_pred_test, y=residuals,
                     title="Residuals Plot",
                     labels={'x': 'Predicted Price ($)', 'y': 'Residuals ($)'})
    fig5.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig5, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and scikit-learn")