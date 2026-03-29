# 🎓 Student Performance Predictor — End-to-End ML Engineering Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green)](https://flask.palletsprojects.com/)
[![AWS](https://img.shields.io/badge/AWS-Elastic%20Beanstalk-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/elasticbeanstalk/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-red)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5.3-150458)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24.3-013243)](https://numpy.org/)

## 🌟 Live Deployment

### 🟧 AWS Elastic Beanstalk (Production): [http://studentperformance-env.eba-nekqq5mg.us-east-1.elasticbeanstalk.com/predictdata](http://studentperformance-env.eba-nekqq5mg.us-east-1.elasticbeanstalk.com/predictdata)

---

## 📖 Project Overview

This project is a **complete, production-ready machine learning system** that predicts a student's math score based on demographic and academic features. It covers the full ML engineering lifecycle — from data ingestion and preprocessing to model training, web application development, and cloud deployment on AWS Elastic Beanstalk.

### 🎯 What This Project Does

- Accepts **7 student features** (demographic + academic) as input
- Automatically selects the **best ML model** from multiple trained algorithms
- Delivers **real-time math score predictions** through a clean web interface
- Fully deployed on **AWS Elastic Beanstalk** with automated setup

---

## 🏗️ Complete Architecture Overview

### 📊 Machine Learning Pipeline

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Raw Data (CSV)    │───▶│   Data Ingestion    │───▶│  Train-Test Split   │
│   - 1000 records    │    │   - Load dataset    │    │  - 80% train        │
│   - 7 features      │    │   - Validate schema │    │  - 20% test         │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                     │                           │
                                     ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Feature Engineering│◀───│ Data Transformation │───▶│   Model Training    │
│  - OneHot Encoding  │    │ - Handle missing    │    │ - Multiple Algos    │
│  - Standard Scaling │    │ - Scale features    │    │ - GridSearchCV      │
│  - Pipeline Creation│    │ - Create artifacts  │    │ - Best Model Select │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                     │                           │
                                     ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Model Artifacts   │◀───│  Model Evaluation   │───▶│   Web Application   │
│   - model.pkl       │    │  - R² Score         │    │   - Flask API       │
│   - preprocessor.pkl│    │  - MAE, RMSE        │    │   - User Interface  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### 🟧 AWS Elastic Beanstalk Deployment Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────────────┐
│   GitHub Repository │───▶│  .ebextensions      │───▶│  AWS Elastic Beanstalk   │
│   - Source Code     │    │  - EB configuration │    │  - Auto-provisioned EC2  │
│   - requirements.txt│    │  - Python platform  │    │  - Load Balancer         │
│   - application.py  │    │  - WSGI setup       │    │  - Auto-scaling          │
└─────────────────────┘    └─────────────────────┘    └──────────────────────────┘
                                                                    │
                                                                    ▼
                                                       ┌──────────────────────────┐
                                                       │    Flask Application     │
                                                       │    - /predictdata route  │
                                                       │    - ML model serving    │
                                                       │    - Real-time inference │
                                                       └──────────────────────────┘
```

---

## 📊 Dataset Analysis & Feature Impact

### 🔍 Dataset Overview

| Property | Value |
|----------|-------|
| **Total Records** | 1,000 student records |
| **Input Features** | 7 (5 categorical + 2 numerical) |
| **Target Variable** | Math Score (0–100, continuous) |
| **Missing Values** | None — clean dataset |
| **Problem Type** | Regression |

### 📈 Feature Impact Analysis

#### 1. Reading Score (Numerical) — Highest Impact
```python
# Correlation with Math Score: ~0.82
# Contribution: ~35-40% of prediction power
```
Strong positive correlation — students who excel at reading tend to perform well in math due to shared cognitive skills like logical reasoning and pattern recognition.

#### 2. Writing Score (Numerical) — Second Highest Impact
```python
# Correlation with Math Score: ~0.80
# Contribution: ~30-35% of prediction power
```
Acts as a proxy for overall academic discipline and study habits. Strong writers typically demonstrate better problem-solving organization.

#### 3. Parental Level of Education (Categorical) — Moderate-High Impact
```python
# Categories: 'some high school', 'high school', 'some college',
#             "associate's degree", "bachelor's degree", "master's degree"
# Contribution: ~15-20% of prediction power
```
Higher parental education correlates with more academic support, resources, and a learning-oriented home environment.

#### 4. Lunch Type (Categorical) — Moderate Impact
```python
# Categories: 'free/reduced', 'standard'
# Contribution: ~10-15% of prediction power
```
Serves as a socioeconomic indicator — students with standard lunch tend to come from higher-income households with fewer financial stressors affecting focus.

#### 5. Test Preparation Course (Categorical) — Moderate Impact
```python
# Categories: 'completed', 'none'
# Contribution: ~8-12% of prediction power
```
Completing a test prep course gives students strategic advantages and reflects family investment in academic success.

#### 6. Race/Ethnicity (Categorical) — Lower Impact
```python
# Categories: 'group A', 'group B', 'group C', 'group D', 'group E'
# Contribution: ~5-8% of prediction power
```
May reflect systemic educational disparities. Handled carefully with one-hot encoding to avoid ordinal bias.

#### 7. Gender (Categorical) — Lowest Direct Impact
```python
# Categories: 'male', 'female'
# Contribution: ~3-5% of prediction power
```
Minimal direct impact but may interact with other features in complex ways.

### 🔬 Statistical Insights

```python
# Math score distribution
# Mean: ~66  |  Std: ~15  |  Range: ~30-100
# Skewness: Slightly left-skewed (more high performers)

# Academic features only → R² = 0.78
# Demographic features only → R² = 0.42
# Combined all features → R² = 0.87–0.91
```

---

## 🤖 Machine Learning Implementation

### 🔧 Preprocessing Pipeline

#### Categorical Encoding
```python
categorical_features = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course"
]
# Strategy: OneHotEncoder — avoids artificial ordinal relationships
```

**Why OneHot over LabelEncoder?**
LabelEncoder assigns integer labels (e.g., group A = 1, group B = 2) implying a false ordering that doesn't exist. OneHotEncoder treats all categories as equal and independent.

#### Feature Scaling
```python
numerical_features = ["reading_score", "writing_score"]
# Strategy: StandardScaler → (x - mean) / std → Mean=0, Std=1
```

**Why StandardScaler?**
Ensures numerical features contribute equally to distance-based algorithms (like KNN) and helps gradient-based models converge faster.

### 🎯 Model Comparison

| Algorithm | R² Score | Training Time | Notes |
|-----------|----------|---------------|-------|
| **Gradient Boosting** | 0.87–0.91 | 2–3 min | Sequential learning, captures non-linearity |
| **Random Forest** | 0.85–0.89 | 1–2 min | Robust ensemble, low overfitting |
| **Linear Regression** | 0.78–0.82 | <1 min | Fast baseline, interpretable |
| **Decision Tree** | 0.75–0.85 | <1 min | Interpretable, prone to overfitting |
| **KNN** | 0.70–0.80 | <1 min | Simple, sensitive to dimensionality |

### 🔬 Hyperparameter Tuning with GridSearchCV

```python
# 5-fold cross-validation across all combinations
# Example — Random Forest search space:
params = {
    'n_estimators': [8, 16, 32, 64, 128, 256],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Total: 6 × 5 × 3 × 3 = 270 combinations tested per model
```

The best model is **automatically selected** based on the highest R² score and saved as `model.pkl`.

---

## 🟧 AWS Elastic Beanstalk Deployment

### 🚀 Live Application
**URL:** [http://studentperformance-env.eba-nekqq5mg.us-east-1.elasticbeanstalk.com/predictdata](http://studentperformance-env.eba-nekqq5mg.us-east-1.elasticbeanstalk.com/predictdata)

### 🏗️ Infrastructure Components

**1. AWS Elastic Beanstalk**
- Platform: Python 3.x on 64-bit Amazon Linux 2
- Auto-provisioned EC2 instance
- Built-in load balancing and auto-scaling
- Health monitoring and log management

**2. `.ebextensions` Configuration**
- Python WSGI configuration pointing to `application:application`
- Dependency installation via `requirements.txt`
- Environment variable management

**3. Application Entry Point**
```python
# application.py — Elastic Beanstalk requires this exact naming
application = Flask(__name__)
app = application

if __name__ == "__main__":
    app.run(host="0.0.0.0")
```

### 📋 Step-by-Step Deployment Guide

#### 1. Prerequisites
```bash
# Install EB CLI
pip install awsebcli

# Configure AWS credentials
aws configure
# Enter: AWS Access Key ID
# Enter: AWS Secret Access Key
# Enter: Default region (e.g., us-east-1)
# Enter: Output format (json)
```

#### 2. Initialize Elastic Beanstalk
```bash
# In your project root directory
eb init -p python-3.8 student-performance-predictor --region us-east-1

# Follow prompts:
# - Select region
# - Create new application
# - Set up SSH (optional)
```

#### 3. Create `.ebextensions` Configuration
```bash
mkdir .ebextensions
```

Create `.ebextensions/python.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: application:application
```

#### 4. Create & Deploy Environment
```bash
# Create environment and deploy
eb create student-performance-env

# For subsequent deployments
eb deploy

# Open the live application
eb open
```

#### 5. Verify Deployment
```bash
# Check environment status
eb status

# View application logs
eb logs

# SSH into EC2 instance (if needed)
eb ssh
```

### 📊 Deployment Metrics

| Metric | Value |
|--------|-------|
| **Deployment Time** | ~3–5 minutes |
| **Cold Start** | <10 seconds |
| **Prediction Response** | <500ms |
| **Platform** | Python on Amazon Linux 2 |
| **Region** | us-east-1 |

### 💰 Cost Optimization

| Resource | Estimated Cost |
|----------|---------------|
| EC2 t2.micro (Free Tier) | $0/month |
| EC2 t2.micro (after free tier) | ~$8.50/month |
| Elastic Beanstalk service | Free |
| Data Transfer | Minimal |

**Tips to save costs:**
- Use `t2.micro` (free tier eligible for 12 months)
- Terminate environments when not in use: `eb terminate`
- Set up budget alerts in AWS Cost Explorer

### 🛠️ Troubleshooting

**Application not starting:**
```bash
# Check logs for errors
eb logs

# Common fix: ensure entry point is named application.py
# and Flask app variable is named 'application'
```

**Dependency errors:**
```bash
# Ensure requirements.txt is in root directory
# Verify all packages are compatible with Python version on EB
pip freeze > requirements.txt
```

**502 Bad Gateway:**
```bash
# Check WSGIPath in .ebextensions/python.config
# Should match: application:application
eb config
```

---

## 🛠️ Project Structure

```
Student-Performance-Predictor/
├── 📁 .ebextensions/
│   └── python.config          # AWS Elastic Beanstalk WSGI config
├── 📁 .vscode/                # VS Code settings
├── 📁 artifacts/
│   ├── model.pkl              # Trained best model
│   ├── preprocessor.pkl       # Fitted data preprocessor
│   └── *.csv                  # Processed train/test datasets
├── 📁 catboost_info/          # CatBoost training logs
├── 📁 notebook/
│   ├── EDA_Student_Performance.ipynb   # Exploratory Data Analysis
│   └── Model_Training.ipynb            # Model experimentation
├── 📁 src/
│   ├── __init__.py
│   ├── exception.py           # Custom exception handling
│   ├── logger.py              # Structured logging
│   ├── utils.py               # Utility functions (save/load objects)
│   ├── 📁 components/
│   │   ├── data_ingestion.py      # Load & split data
│   │   ├── data_transformation.py # Feature engineering & preprocessing
│   │   └── model_trainer.py       # Train, evaluate & select best model
│   └── 📁 pipeline/
│       ├── predict_pipeline.py    # Inference pipeline (CustomData + PredictPipeline)
│       └── train_pipeline.py      # End-to-end training pipeline
├── 📁 templates/
│   ├── index.html             # Landing page
│   └── home.html              # Prediction form & results
├── application.py             # Flask app (EB-compatible entry point)
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── .gitignore
└── README.md
```

---

## 📦 Tech Stack & Dependencies

```txt
Flask==2.3.2          # Web framework
numpy==1.24.3         # Numerical computing
pandas==1.5.3         # Data manipulation
scikit-learn==1.2.2   # ML algorithms & preprocessing
seaborn==0.12.2       # Statistical visualization
dill==0.3.6           # Model serialization (extended pickle)
```

---

## 💻 Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/krotov22/Student-Performance-Predictor.git
cd Student-Performance-Predictor
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline
```bash
python src/components/data_ingestion.py
```
This will trigger the full pipeline: data ingestion → transformation → model training → artifact saving.

### 5. Start the Web Application
```bash
python application.py
```

### 6. Open in Browser
```
http://localhost:5000/predictdata
```

---

## 📱 Web Application

### Features
- **Prediction Form** — Input 7 student features via a clean, responsive UI
- **Real-time Results** — Math score prediction returned in under 500ms
- **Input Validation** — Form handles edge cases gracefully
- **Mobile-friendly** — Responsive layout for all screen sizes

### API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Landing/index page |
| `/predictdata` | GET | Render the prediction form |
| `/predictdata` | POST | Submit features, return predicted score |

### Prediction Request Flow
```
User Input (form) 
    → CustomData object (predict_pipeline.py)
    → DataFrame conversion
    → preprocessor.pkl (transform features)
    → model.pkl (predict score)
    → Rendered result on home.html
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Best R² Score** | ~0.88–0.91 |
| **Mean Absolute Error** | ~5.8 points |
| **Root Mean Square Error** | ~7.6 points |

The system **automatically selects** the best model during training — whichever algorithm achieves the highest R² on the test set is saved and used for production inference.

### Feature Importance Summary
1. **Reading Score** — Strongest predictor (~35–40%)
2. **Writing Score** — Second strongest (~30–35%)
3. **Parental Education** — Key socioeconomic factor (~15–20%)
4. **Test Preparation** — Completion provides notable advantage (~8–12%)
5. **Lunch Type** — Socioeconomic proxy (~10–15%)

---

## 🔍 Key Learnings & Challenges

### What I Built From Scratch
- **Modular ML pipeline** — separate, reusable components for ingestion, transformation, and training
- **Custom exception & logging** — production-grade error tracking
- **Automated model selection** — system picks the best algorithm without human intervention
- **AWS Elastic Beanstalk deployment** — full cloud deployment with proper WSGI configuration

### Challenges Solved

**1. EB Entry Point Naming**
Elastic Beanstalk requires the Flask app to be in a file named `application.py` and the Flask instance must be named `application`. Took time to debug the 502 error caused by incorrect WSGI path configuration.

**2. Pickle Compatibility**
Models serialized with one Python version failed to load on another. Solved by standardizing the Python version and using `dill` for more robust serialization.

**3. Feature Leakage**
Initially included target-correlated features by mistake. Learned the importance of strict train/test separation and proper pipeline design.

**4. Regression vs Classification**
Started confused about problem type. Understood that predicting a continuous score (0–100) is a regression problem, not classification — and that R², MAE, and RMSE are the right evaluation metrics.

---

## 🚀 Future Enhancements

- [ ] Add **GitHub Actions CI/CD** pipeline for automated EB deployments on push
- [ ] Integrate **Docker** containerization for consistent environments
- [ ] Explore **XGBoost & CatBoost** with advanced hyperparameter tuning
- [ ] Add **model monitoring** — track prediction drift over time
- [ ] Build an **analytics dashboard** with Plotly/Dash for EDA visualization
- [ ] Add **user authentication** and prediction history storage
- [ ] Implement **Bayesian hyperparameter optimization** (Optuna)
- [ ] Add **AWS RDS** integration for persistent data storage

---

## 📞 Contact

<div align="center">

### 👨‍💻 krotov22

[![Email](https://img.shields.io/badge/Email-bhikams468%40gmail.com-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:bhikams468@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Satyansh%20Singh-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/satyansh-singh-567a22260)
[![GitHub](https://img.shields.io/badge/GitHub-krotov22-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/krotov22)

---

### 🌟 Show Your Support

If you found this project helpful, please give it a ⭐!

[![GitHub stars](https://img.shields.io/github/stars/krotov22/Student-Performance-Predictor?style=social)](https://github.com/krotov22/Student-Performance-Predictor)
[![GitHub forks](https://img.shields.io/github/forks/krotov22/Student-Performance-Predictor?style=social)](https://github.com/krotov22/Student-Performance-Predictor)

</div>

---

<div align="center">

*This project demonstrates end-to-end ML engineering — from raw data to a live, cloud-deployed prediction system.*

**🤖 ML Engineering • 🟧 AWS Cloud Deployment • 🐍 Python • 🌐 Flask Web App**

### 🏆 Key Achievements
- ✅ **Live Production App** on AWS Elastic Beanstalk
- ✅ **Automated Model Selection** with GridSearchCV
- ✅ **91% R² Score** on best performing model
- ✅ **Modular Pipeline Architecture** — ingestion → transformation → training
- ✅ **Sub-500ms Prediction Response** in production
- ✅ **Clean Web Interface** with real-time inference

</div>
