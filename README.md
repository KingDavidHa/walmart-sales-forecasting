# Walmart Sales Forecasting Project

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/AWS-SageMaker-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.3-green.svg)

## Project Overview

This project implements an end-to-end machine learning solution for forecasting Walmart's weekly sales by store. The model utilizes historical sales data combined with features like holidays, temperature, fuel prices, CPI, and unemployment rates to predict future sales patterns.

### Key Results
- **R² Score**: 0.9919 (explains 99.19% of sales variance)
- **RMSE**: 49,666.76
- **Mean Absolute Error**: 33,750.63
- **MAPE**: ~5.99%

## Dataset Description

The analysis is based on historical Walmart sales data with the following characteristics:

- **Size**: 6,435 rows, 8 columns
- **Stores**: 45 unique stores
- **Key Variables**: Store number, Weekly Sales, Holiday Flag, Temperature, Fuel Price, CPI, Unemployment Rate
- **Time Range**: Multiple weeks of historical sales data

```
Dataset structure:
Store Date Weekly_Sales Holiday_Flag Temperature Fuel_Price CPI Unemployment
```

## Key Features & Insights

### Sales Pattern Analysis
- **Seasonality**: Peak sales occur in November and December (holiday shopping season)
- **Store Variability**: Significant differences in sales patterns across different stores
- **Holiday Effect**: Sales increase during holiday periods (450 instances with Holiday Flag=1)

### Key Correlations Found
- Store location and month have strong correlations with sales performance
- Date-based features (year, month, week of year) are important predictors
- Environmental variables (temperature, fuel price) influence purchasing behavior

## Model Development Approach

### Feature Engineering
1. **Temporal Feature Extraction**
   - Extracted year, month, day, day of week, week of year from dates
   - Utilized holiday flags to capture seasonal patterns

2. **Lag Features Creation**
   - Leveraged sales data from 1, 2, and 3 weeks prior
   - Calculated 4-week rolling mean
   - These features significantly improved forecasting accuracy

### Machine Learning Model
- **Algorithm**: XGBoost regression model (effective for capturing non-linear patterns)
- **Hyperparameters**:
  ```
  n_estimators=100
  learning_rate=0.1
  max_depth=5
  subsample=0.8
  colsample_bytree=0.8
  ```
- **Validation**: 80/20 train/test split

### AWS Deployment
- Model deployed using AWS SageMaker
- Training compute: ml.m5.large instance
- Inference resources: ml.m5.large instance
- Configured endpoint with dynamic scaling capabilities

## Feature Importance

The most important predictors for sales forecasting:

1. 4-week rolling mean (Sales_Roll_Mean_4): 521.0
2. Day: 306.0
3. Week of Year (WeekOfYear): 304.0
4. 1-week prior sales (Sales_Lag_1): 281.0
5. 3-weeks prior sales (Sales_Lag_3): 268.0

This indicates that recent sales history and calendar features are critical for accurate predictions.

## Business Applications

### Short-term Value
1. **Inventory Optimization**: Adjust inventory levels based on predicted store sales
2. **Workforce Planning**: Optimize staff allocation based on sales forecasts
3. **Promotion Analytics**: Measure promotion effectiveness against baseline predictions

### Long-term Strategic Value
1. **Store Expansion Strategy**: Inform new location selection based on sales patterns
2. **Category Optimization**: Recommend specialized product categories by store
3. **Risk Management**: Identify stores with high sales volatility and develop strategies

## Getting Started

### Prerequisites
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- AWS SageMaker (for cloud deployment)

### Installation
```bash
# Clone this repository
git clone https://github.com/yourusername/walmart-sales-forecasting.git

# Change directory
cd walmart-sales-forecasting

# Install required packages
pip install -r requirements.txt
```

### Usage Example
```python
# Load the trained model
import pickle

with open('walmart_sales_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
# Prepare future data for prediction
future_data = prepare_future_data(last_date, num_weeks=8, store_ids=[1, 2, 3])

# Make predictions
predictions = model.predict(future_data)
```

## Future Enhancements

### Model Improvements
1. **Advanced Time Series Models**
   - Implement LSTM or Prophet models
   - Explore hierarchical modeling (store → department → category)

2. **External Data Integration**
   - Incorporate local events, competitor promotions, consumer sentiment
   - Add social media trends and search traffic data

### Technical Infrastructure
1. **Real-time Prediction System**
   - Build prediction pipelines updated daily or in real-time
   - Implement automated model retraining and deployment

2. **Business Intelligence Integration**
   - Integrate predictions into management dashboards
   - Develop anomaly detection and alert systems

## Conclusion

This project demonstrates how machine learning can significantly improve sales forecasting accuracy in retail. With a model explaining over 99% of the variance, retailers can make more data-driven decisions, leading to improved operational efficiency and increased profitability.

The implementation provides an end-to-end framework that can be extended to other retail business domains.

