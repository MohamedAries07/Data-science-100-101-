# Data-science-100-101-
�

🌱 CarbonTrack
Predicting Household Carbon Footprint Using Lifestyle Data
�
�
�
�
�
�
Load image
Load image
Load image
Load image
Load image
Load image
Mini Project | CSE - AIML | SRM Institute of Science and Technology
�

📌 Project Overview
CarbonTrack is a machine learning-powered data science project designed to predict the carbon footprint of households based on their daily lifestyle habits. By analysing factors such as electricity consumption, fuel usage, transportation patterns, food habits, and waste generation, this project builds predictive models that can help individuals and policymakers understand and reduce their environmental impact.
📝 Abstract
Climate change driven by greenhouse gas emissions remains one of the most pressing global challenges. Household activities contribute significantly to carbon emissions through electricity consumption, transportation, food choices, and waste generation. This project, CarbonTrack, aims to predict the household carbon footprint score using lifestyle-based features collected from survey data. A dataset of 1,500 household records with 10 key features including electricity usage, monthly fuel consumption, vehicle type, daily travel distance, food habits, water usage, waste generation, family size, and income level was used. Exploratory Data Analysis (EDA) was performed to understand data distributions and inter-feature correlations. After data cleaning, encoding, and feature scaling, two machine learning models — Random Forest Regressor and Linear Regression — were trained and evaluated. The Random Forest model achieved an R² score of 0.91, MAE of 18.4, and RMSE of 24.7, significantly outperforming Linear Regression. The results indicate that electricity consumption, fuel usage, and vehicle type are the strongest predictors of carbon footprint. This project demonstrates the practical application of data science in environmental sustainability.
❗ Problem Statement
Despite growing awareness of climate change, most households lack personalised, data-driven insights into their individual carbon footprints. Existing calculators are simplistic and do not capture the complex, non-linear relationships between lifestyle variables and actual carbon emissions. There is a critical need for an intelligent, machine learning-based system that can accurately predict household carbon emissions from lifestyle data, enabling targeted behaviour change and supporting global emission reduction goals.
🎯 Objectives
Collect and preprocess a realistic household carbon footprint dataset
Perform comprehensive Exploratory Data Analysis (EDA)
Engineer meaningful features from raw lifestyle data
Develop and compare machine learning regression models
Identify the most impactful lifestyle factors on carbon emissions
Visualise findings through informative, publication-ready charts
Provide actionable insights for carbon footprint reduction
📊 Dataset Details
Attribute
Details
Source
UCI ML Repository / Kaggle (Carbon Footprint Dataset)
Records
1,500 household entries
Features
10 input features + 1 target variable
Format
CSV
Missing Values
~3.2% (handled via median imputation)
Feature Description
Feature
Type
Description
electricity_kwh
Numerical
Monthly electricity consumption (kWh)
fuel_liters
Numerical
Monthly fuel usage (litres)
vehicle_type
Categorical
Car type: EV / Hybrid / Petrol / Diesel / None
travel_km_day
Numerical
Average daily travel distance (km)
food_habit
Categorical
Diet type: Vegan / Vegetarian / Mixed / Meat-heavy
water_liters_day
Numerical
Daily water usage (litres)
waste_kg_week
Numerical
Weekly waste generation (kg)
family_size
Numerical
Number of household members
income_level
Categorical
Low / Middle / High
carbon_score
Numerical
Target: Carbon emission score (kg CO₂/month)
🛠️ Technologies Used
Category
Tools
Language
Python 3.10+
Data Manipulation
Pandas, NumPy
Visualisation
Matplotlib, Seaborn
Machine Learning
Scikit-learn
Notebook
Jupyter Notebook
Version Control
Git, GitHub
IDE
VS Code / JupyterLab
🔄 Methodology / Workflow
Problem Identification
        ↓
Dataset Collection & Loading
        ↓
Data Cleaning & Preprocessing
  (Null Handling → Encoding → Scaling)
        ↓
Exploratory Data Analysis (EDA)
  (Distributions → Correlations → Outliers)
        ↓
Data Visualisation
  (Heatmap, Histogram, Scatter, Bar, Pie)
        ↓
Model Development
  (Train-Test Split → Random Forest → Linear Regression)
        ↓
Model Evaluation
  (MAE, RMSE, R² Score, Feature Importance)
        ↓
Result Interpretation & Reporting
⚙️ Installation
# 1. Clone the repository
git clone https://github.com/MohamedAries/MiniProject_DS_AIML-B_2026_CarbonTrack.git

# 2. Navigate to the project folder
cd MiniProject_DS_AIML-B_2026_CarbonTrack

# 3. Install dependencies
pip install -r requirements.txt
▶️ Execution
# Run preprocessing
python src/preprocessing.py

# Run analysis and visualisation
python src/analysis.py

# Train and evaluate models
python src/model.py

# Or open Jupyter Notebooks
jupyter notebook notebooks/data_understanding.ipynb
📁 Folder Structure
MiniProject_DS_AIML-B_2026_CarbonTrack/
│
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
│
├── docs/
│   ├── abstract.pdf                 # Project abstract
│   ├── problem_statement.pdf        # Detailed problem statement
│   └── presentation.pptx           # Project presentation slides
│
├── dataset/
│   ├── raw_data/                    # Original unprocessed dataset
│   └── processed_data/             # Cleaned and encoded dataset
│
├── notebooks/
│   ├── data_understanding.ipynb    # EDA and data profiling
│   ├── preprocessing.ipynb         # Cleaning and transformation
│   └── visualization.ipynb         # Charts and graphs
│
├── src/
│   ├── preprocessing.py            # Data preprocessing script
│   ├── analysis.py                 # EDA and visualisation script
│   └── model.py                    # ML model training and evaluation
│
├── outputs/
│   ├── graphs/                     # Saved visualisation outputs
│   └── results/                    # Model results and metrics
│
└── report/
    └── mini_project_report.pdf     # Final project report
📈 Sample Results
Model
MAE
RMSE
R² Score
Random Forest Regressor
18.4
24.7
0.91
Linear Regression
31.2
42.5
0.73
Top 5 Feature Importances (Random Forest)
Rank
Feature
Importance
1
electricity_kwh
0.28
2
fuel_liters
0.24
3
vehicle_type
0.18
4
travel_km_day
0.14
5
food_habit
0.09
🔮 Future Scope
Integration of real-time smart meter data via IoT sensors
Development of a web application dashboard for household users
Incorporation of regional climate and energy mix data
Extension to community-level and city-level carbon tracking
Deployment as a mobile app with personalised recommendations
👤 Team Members
Name
Register Number
Role
Mohamed Aries B
RA2311026050100
Project Lead & Developer
Department: CSE - Artificial Intelligence and Machine Learning (AIML-B)
Institution: SRM Institute of Science and Technology
Academic Year: 2025–2026
📜 License
This project is licensed under the MIT License — see the LICENSE file for details.
