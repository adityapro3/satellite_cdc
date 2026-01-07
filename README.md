# Satellite-Based House Price Prediction

## 🎯 Project Overview

This project leverages **satellite imagery** combined with **tabular features** to predict house prices in King County, Washington. By integrating aerial NAIP (National Agriculture Imagery Program) imagery with property characteristics, the model achieves superior predictive performance.

### Key Highlights
- **Final RMSE**: $104,710
- **R² Score**: 0.9069 (90.69% variance explained)
- **Architecture**: XGBoost + ResNet50 feature extraction
- **Dataset**: 16,209 house sales records
- **Data Source**: King County, Washington (2014-2015)

---

## 📁 Project Structure

```
├── data_fetcher.py              # Google Earth Engine data collection
├── Classification.ipynb       # Exploratory Data Analysis
├── modelling.ipynb           # Model training and evaluation
├── xgb_model.json              # Trained XGBoost model
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

---

## 🔧 Installation & Setup

### Prerequisites
```bash
Python 3.8+
Google Earth Engine account (for data fetching)
```

### Dependencies
```bash
pip install pandas numpy scikit-learn xgboost
pip install opencv-python rasterio geemap earthengine-api
pip install torch timm matplotlib seaborn tqdm
```

### Google Earth Engine Setup
```python
# Authenticate GEE
import ee
service_account = 'your-service-account@project.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'key-file.json')
ee.Initialize(credentials)
```

---

## 📊 Data Pipeline

### 1. **Data Fetching** (`data_fetcher.py`)

Downloads NAIP satellite imagery for property locations:

**Key Features:**
- Multi-threaded download (6 parallel workers)
- 80m buffer around each property
- RGB bands at 1m resolution
- Automatic checkpoint/resume capability

**Usage:**
```python
python data_fetcher.py
```

**Output:**
- RGB satellite images (224x224 pixels)
- Progress tracking with checkpoint system
- ~16,000 images downloaded

---

### 2. **Exploratory Data Analysis** (`Classification.ipynb`)

Comprehensive analysis including:

#### Key Findings:
- **Price Distribution**: Right-skewed (skewness: 4.03)
- **Top Predictor**: sqft_living (r = 0.70)
- **Waterfront Premium**: 130% above median
- **Geographic Clustering**: Strong lat/long correlations (0.3-0.4)

#### Visualizations:
- Distribution plots for all features
- Correlation heatmaps
- Box plots for outlier detection
- Satellite image samples with property info
- Price distributions by location

#### Data Quality:
- ✅ No missing values
- ✅ No duplicate records
- ✅ Outliers handled (price quantile filtering)
- ✅ Feature engineering (house_age, is_renovated, lat_long_interaction)

---

### 3. **Modeling** (`modelling.ipynb`)

#### Model Architecture

**Baseline Model: Tabular-Only XGBoost**
```python
XGBRegressor(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8
)
```
**Results:** RMSE: 109,630 | R²: 0.8929

---

**Final Model: Tabular + Image Features**

**Image Feature Extraction:**
1. **Pre-trained ResNet50** (ImageNet weights)
   - Deep learning feature extraction
   - 2048-dimensional embeddings

2. **Custom Image Features** (13 features):
   ```python
   - green_ratio: Vegetation coverage
   - water_ratio: Water body presence
   - edge_density: Structural complexity
   - brightness_mean: Overall lighting
   - texture_lap_var: Texture variation
   - glcm_contrast: GLCM contrast
   - glcm_homogeneity: GLCM homogeneity
   - glcm_energy: GLCM energy
   - edge_orient_entropy: Edge orientation entropy
   - road_proxy: Road/infrastructure indicator
   - green_spread: Vegetation distribution
   - shadow_ratio: Shadow coverage
   - color_entropy: Color diversity
   ```

**Final XGBoost Configuration:**
```python
XGBRegressor(
    n_estimators=2500,
    max_depth=6,
    learning_rate=0.02,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1
)
```

**Final Results:** 
- **RMSE: $104,710** (4.5% improvement)
- **R² Score: 0.9069** (90.69% variance explained)

---

#### Grad-CAM Visualization

**Purpose:** Understand which image regions influence price predictions

**Key Insights:**
- Crowded areas → Higher prices
- Vegetation presence → Price variation
- Structured housing → Higher valuations
- Neighborhood density impacts predictions

**Implementation:**
```python
def grad_cam(img_path, model):
    # Load and preprocess image
    # Generate activation maps
    # Overlay heatmap on original image
    # Visualize important regions
```

---

## 🎯 Model Performance

| Model | RMSE ($) | R² Score | Features |
|-------|---------|----------|----------|
| XGBoost (Tabular) | 109,630 | 0.8929 | 22 |
| **XGBoost (Tabular + Image)** | **104,710** | **0.9069** | **36** |

### Feature Importance (Top 10)

```
1. grade              : 28.18
2. sqft_living        : 10.48
3. lat                : 7.47
4. lat_long_interaction: 2.75
5. sqft_living15      : 2.07
6. waterfront         : 1.62
7. view               : 1.65
8. long               : 1.22
9. house_age          : 1.03
10. bathrooms         : 0.92
```

---

## 🚀 Usage

### Training the Model

```python
# Load data
df = pd.read_csv('train_processed.csv')

# Extract image features
from modeling import extract_image_features
df_enhanced = add_image_features(df, 'satellite_images/')

# Train model
from xgboost import XGBRegressor
model = XGBRegressor(**params)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
```

### Making Predictions

```python
# Load trained model
import xgboost as xgb
model = xgb.Booster()
model.load_model('xgb_model.json')

# Prepare features
features = prepare_features(property_data, satellite_image)

# Predict
price = np.expm1(model.predict(features))
print(f"Predicted Price: ${price:,.2f}")
```

---

## 📚 Technical Stack

| Component | Technology |
|-----------|------------|
| Data Collection | Google Earth Engine, NAIP |
| Image Processing | OpenCV, Rasterio |
| Deep Learning | PyTorch, Timm |
| ML Framework | XGBoost, Scikit-learn |
| Visualization | Matplotlib, Seaborn, Grad-CAM |
| Data Analysis | Pandas, NumPy |

---


## 📄 License

This project is for educational and research purposes.

---

## 👥 Contributors

**Aditya Sharma** - Data Science & Machine Learning

---
