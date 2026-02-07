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
# 1. Categorical Encoding (One-Hot)
categorical_features = ['gender', 'marital_status', 'ethnicity']
encoded = pd.get_dummies(categorical_features, drop_first=True)

# 2. Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
diagnosis_features = vectorizer.fit_transform(diagnosis_text)

# 3. Numerical Scaling (StandardScaler)
scaler = StandardScaler()
age_scaled = scaler.fit_transform(age)
vitals_scaled = scaler.fit_transform(vitals_df)

# 4. Feature Concatenation (Sparse Matrix)
X = hstack([age_scaled, encoded, vitals_scaled, diagnosis_features])
```

### 4. Model Development (In Progress)

**Planned Approaches**:
- **Supervised Learning**: Logistic Regression, Random Forest, Gradient Boosting
- **Handling Class Imbalance**: 
  - Class weighting
  - SMOTE (Synthetic Minority Over-sampling)
  - Stratified sampling
- **Advanced Models**: XGBoost, LightGBM (native missing value handling)

### 5. Evaluation & Visualization (Planned)
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices
- Feature importance analysis
- Learning curves

## 🚀 Key Technical Achievements

### Performance Optimization
- **Vectorized Operations**: Replaced 58,976-iteration loop with pandas `groupby` + `merge` operations
  - Reduced vitals processing time from **40+ minutes → 3-5 minutes** (8-13x speedup)
- **Memory-Efficient Chunking**: Processed 30GB CHARTEVENTS file in 500K-row chunks
- **Data Persistence**: Serialized preprocessed data to CSV for instant reloading

### Code Quality
- Modular pipeline with clear separation of concerns (ingestion → preprocessing → modeling)
- Defensive programming with file existence checks and error handling
- Comprehensive inline documentation
- Reproducible workflow with fixed random seeds

### Data Engineering
- Multi-table joins across 4+ CSV sources
- Temporal windowing (24-hour clinical data extraction)
- Handling anonymized/shifted timestamps in MIMIC-III
- Safe deserialization of nested data structures (vitals as arrays)

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
- [x] Feature encoding and scaling
- [x] Data persistence (CSV serialization/deserialization)

**In Progress 🔨**
- [ ] Train/test split with stratification
- [ ] Model training (multiple algorithms)
- [ ] Hyperparameter tuning
- [ ] Cross-validation

**Planned 📋**
- [ ] Model evaluation and metrics
- [ ] Feature importance analysis
- [ ] Result visualization
- [ ] Deployment-ready inference pipeline
- [ ] Integration with pediatric ICU dataset (next phase)

## 💡 Future Enhancements

1. **Real-Time Prediction**: Adapt retrospective model for prospective use with streaming data
2. **Explainable AI**: SHAP values for clinical interpretability
3. **Multi-Task Learning**: Predict ICU length of stay, mortality, and admission simultaneously
4. **External Validation**: Test on Children's Hospital dataset
5. **MLOps Pipeline**: Containerization, CI/CD, model versioning

## 📚 Learning Outcomes

This project demonstrates proficiency in:
- Large-scale data processing (40GB+ datasets)
- Healthcare data standards (MIMIC-III schema)
- Feature engineering for time-series clinical data
- Handling severe class imbalance
- Performance optimization (vectorization, chunking)
- End-to-end ML pipeline development

## 📞 Contact

**Griffin Kuchar**  
griffin.kuchar@gmail.com | https://www.linkedin.com/in/griffin-kuchar-95081124b/ | https://github.com/gkuchar

---

*This project was developed as part of collaborative research on ICU admission prediction models. The pipeline is designed to be adaptable to other clinical datasets for broader healthcare applications.*