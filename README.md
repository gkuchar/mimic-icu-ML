# ICU Admission Prediction Model

A machine learning pipeline for predicting ICU admissions using clinical data from the MIMIC-III Critical Care Database.

## 🎯 Project Overview

This project develops a predictive model to identify patients who require ICU admission based on their initial clinical presentation and vital signs within the first 24 hours of hospital admission. The model processes multi-modal clinical data including demographics, diagnosis text, vital signs, and patient history to support clinical decision-making.

**Primary Use Case**: Real-time assessment of patient acuity to optimize ICU resource allocation and improve patient outcomes.

## 📊 Dataset

- **Source**: [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) (58,976 hospital admissions)
- **Access**: Credentialed researcher with approved PhysioNet account
- **Size**: ~40GB of clinical records across multiple CSV files
- **Target Variable**: ICU admission (binary classification)
  - Class distribution: 98% admitted to ICU, 2% not admitted
  - Note: MIMIC-III is ICU-centric by design

## 🔧 Technical Stack

- **Language**: Python 3.12
- **Environment**: Google Colab (cloud-based Jupyter notebooks)
- **Core Libraries**:
  - `pandas`, `numpy` - Data manipulation and numerical computing
  - `scikit-learn` - Machine learning models and preprocessing
  - `scipy` - Sparse matrix operations for high-dimensional features
  - `matplotlib`, `seaborn` - Data visualization
  - `ast` - Safe parsing of serialized Python structures

## 🏗️ Architecture & Pipeline

### 1. Data Ingestion
Multi-source data extraction from MIMIC-III CSV files:

| Data Source | Fields Extracted | Purpose |
|------------|------------------|---------|
| `ADMISSIONS.csv` | `HADM_ID`, `SUBJECT_ID`, `ETHNICITY`, `DIAGNOSIS`, `MARITAL_STATUS`, `ADMITTIME` | Patient demographics and admission details |
| `PATIENTS.csv` | `GENDER`, `DOB` | Patient characteristics |
| `ICUSTAYS.csv` | `HADM_ID` | Target variable (ICU admission) |
| `CHARTEVENTS.csv` | Vital signs (HR, BP, Temp, RR, SpO2) | Clinical measurements (first 24 hours) |

**Key Design Decision**: Using `HADM_ID` (hospital admission ID) as the primary key rather than `SUBJECT_ID` (patient ID) to treat each hospital visit as an independent sample and prevent data leakage from multiple admissions.

### 2. Feature Engineering

#### Temporal Feature Extraction
```python
# Calculate age at admission (handles MIMIC's anonymized dates)
age = (admission_datetime - dob_datetime).days // 365

# Extract first 24-hour vitals window
vitals_24hr = vitals[(vitals['CHARTTIME'] >= admittime) & 
                     (vitals['CHARTTIME'] <= admittime + timedelta(hours=24))]
```

#### Vitals Aggregation Strategy
Optimized vectorized approach for processing 30GB+ CHARTEVENTS file:
- Chunked reading (500K rows/chunk) to manage memory
- Filtered by relevant ITEMID codes before full load
- Merged admission timestamps for efficient windowing
- Grouped aggregation (mean) per vital sign type per admission
- **Performance**: ~3-5 minutes vs. 40+ minutes with naive row-by-row iteration

**Vital Signs Tracked**:
- Heart Rate (HR)
- Systolic/Diastolic Blood Pressure (SBP/DBP)
- Temperature
- Respiratory Rate (RR)
- Oxygen Saturation (SpO2)

### 3. Data Preprocessing

#### Handling Missing Data
- **Diagnosis**: Dropped rows with missing values (25 rows, <0.05%)
- **Marital Status**: Imputed with "Not Disclosed" category
- **Vitals**: Median imputation using `SimpleImputer`

#### Feature Transformation Pipeline
```python
# ColumnTransformer with stratified pipelines
preprocessor = ColumnTransformer([
    # 1. Vitals: Impute → Scale
    ('vitals', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), vitals_columns),
    
    # 2. Age: Scale
    ('age', StandardScaler(), ['age']),
    
    # 3. Categorical: One-Hot Encode
    ('cat', OneHotEncoder(handle_unknown='ignore'), 
     ['gender', 'marital_status', 'ethnicity']),
    
    # 4. Text: TF-IDF Vectorize
    ('text', TfidfVectorizer(max_features=100, stop_words='english'), 
     'diagnosis')
])
```

### 4. Model Development & Training

**Models Evaluated**:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Naive Bayes
- K-Nearest Neighbors (KNN)

**Training Configuration**:
- Train/test split: 80/20 with stratification
- Random state: 0 (reproducibility)
- Hyperparameters optimized for each model

### 5. Results & Model Performance

#### Top Performing Models

| Model | ROC-AUC | PR-AUC | Specificity | Training Time |
|-------|---------|---------|-------------|---------------|
| **Gradient Boosting** | **0.927** | **0.998** | 0.105 | 25.8s |
| Random Forest | 0.869 | 0.996 | **0.152** | 5.9s |
| Logistic Regression | 0.870 | 0.996 | 0.000 | 0.9s |
| SVM | 0.843 | 0.996 | 0.000 | 194.0s |
| KNN | 0.692 | 0.988 | 0.076 | 0.004s |
| Naive Bayes | 0.854 | 0.995 | 0.928 | 0.058s |

**Winner: Gradient Boosting**
- **Best ROC-AUC**: 0.927 - Superior ability to distinguish ICU vs non-ICU patients
- **Excellent PR-AUC**: 0.998 - Maintains precision-recall balance despite severe class imbalance
- **F1 Score**: 0.990 - Strong overall performance
- **Trade-off**: Slower training (26s) but acceptable for this use case

#### Key Findings

**Class Imbalance Challenge**:
The 98/2 ICU distribution posed significant challenges:
- Most models achieved 0% specificity (predicted ICU for all patients)
- Naive Bayes showed highest specificity (92.8%) but poor recall (36.4% - dangerous for missing ICU patients)
- Gradient Boosting achieved best balance with 10.5% specificity while maintaining 99.9% recall

**Model Insights**:
1. **Gradient Boosting**: Best overall - captures non-linear patterns in vitals and diagnosis
2. **Random Forest**: Best specificity among competitive models - good for identifying truly non-ICU cases
3. **Logistic Regression**: Fast baseline but limited by linear assumptions
4. **SVM**: Too slow (194s training) for marginal performance gain
5. **KNN**: Poor generalization (ROC-AUC: 0.692) - doesn't scale well to high-dimensional sparse features
6. **Naive Bayes**: High specificity but misses 64% of ICU patients - unacceptable for clinical use

**Confusion Matrix Analysis**:
- Gradient Boosting: 25 FP, 12 FN - Balanced error distribution
- Random Forest: 36 FP, 58 FN - More conservative, higher false negatives
- Naive Bayes: 220 FP, 7345 FN - Too conservative, misses most ICU cases

### 6. Visualization

Comprehensive evaluation includes:
- **Confusion matrices** for all models (visual comparison of prediction patterns)
- **Metrics dashboard** with 9 performance indicators
- **Model comparison charts** showing trade-offs between speed and accuracy

## 🚀 Key Technical Achievements

### Performance Optimization
- **Vectorized Operations**: Replaced 58,976-iteration loop with pandas `groupby` + `merge` operations
  - Reduced vitals processing time from **40+ minutes → 3-5 minutes** (8-13x speedup)
- **Memory-Efficient Chunking**: Processed 30GB CHARTEVENTS file in 500K-row chunks
- **Data Persistence**: Serialized preprocessed data to CSV for instant reloading

### Code Quality
- Modular pipeline with clear separation of concerns (ingestion → preprocessing → modeling)
- Production-ready preprocessing with `ColumnTransformer` and `Pipeline`
- Defensive programming with file existence checks and error handling
- Comprehensive inline documentation
- Reproducible workflow with fixed random seeds

### Data Engineering
- Multi-table joins across 4+ CSV sources
- Temporal windowing (24-hour clinical data extraction)
- Handling anonymized/shifted timestamps in MIMIC-III
- Safe deserialization of nested data structures (vitals as arrays)

### Machine Learning Best Practices
- **Train/test split before preprocessing** to prevent data leakage
- **Stratified sampling** to maintain class distribution
- **Comprehensive metrics** beyond accuracy for imbalanced data
- **Model comparison** across speed/accuracy trade-offs
- **Visual evaluation** with confusion matrices

## 📁 Project Structure

```
.
├── mimicICU.ipynb                 # Main pipeline notebook
├── mimic_complete_data.csv        # Preprocessed dataset (58,951 rows × 9 features)
├── README.md                      # This file
├── .gitignore
└── data/                          # Raw MIMIC-III CSV files (not included)
    ├── ADMISSIONS.csv
    ├── PATIENTS.csv
    ├── ICUSTAYS.csv
    └── CHARTEVENTS.csv
```

## 🔄 Current Status

**Completed ✅**
- [x] PhysioNet credentialing and MIMIC-III access
- [x] Multi-source data ingestion pipeline
- [x] Feature engineering (demographics, vitals, diagnosis text)
- [x] Missing data handling
- [x] Production-ready preprocessing pipeline
- [x] Train/test split with stratification
- [x] Model training (6 algorithms)
- [x] Comprehensive evaluation metrics
- [x] Confusion matrix visualization
- [x] Data persistence (CSV serialization/deserialization)

## 📚 Learning Outcomes

This project demonstrates proficiency in:
- **Large-scale data processing** (40GB+ datasets with chunking strategies)
- **Healthcare data standards** (MIMIC-III schema, clinical terminology)
- **Feature engineering** for time-series clinical data with temporal windowing
- **Handling severe class imbalance** (98/2 split with appropriate metrics)
- **Performance optimization** (vectorization, 8-13x speedup)
- **Production ML pipelines** (ColumnTransformer, Pipeline, proper train/test splitting)
- **Model selection** across diverse algorithms (tree-based, linear, probabilistic, instance-based)
- **Evaluation for imbalanced data** (ROC-AUC, PR-AUC, specificity)
- **Data visualization** for model comparison and clinical interpretation

## 🎓 Key Insights for Recruiters

**Problem-Solving Approach**:
- Identified class imbalance as primary challenge and selected metrics accordingly (ROC-AUC, PR-AUC over accuracy)
- Optimized data pipeline (8-13x speedup) through algorithm analysis
- Chose appropriate model architecture (Gradient Boosting) over complex deep learning

**Technical Depth**:
- End-to-end pipeline from raw 40GB clinical data → trained models
- Production-ready code with proper preprocessing, no data leakage
- Comprehensive evaluation beyond surface-level metrics

**Business Impact**:
- 92.7% ROC-AUC enables effective patient risk stratification
- 99.9% recall ensures minimal missed ICU cases (critical for patient safety)
- Fast inference (0.028s) suitable for real-time clinical decision support

## 📞 Contact

**Griffin Kuchar**  
📧 griffin.kuchar@gmail.com  
💼 [LinkedIn](https://www.linkedin.com/in/griffin-kuchar-95081124b/)  
💻 [GitHub](https://github.com/gkuchar)

---

*This project was developed as part of collaborative research on ICU admission prediction models. The pipeline is designed to be adaptable to other clinical datasets for broader healthcare applications.*