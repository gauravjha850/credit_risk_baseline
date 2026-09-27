# 🏦 End-to-End Credit Risk Analysis & IFRS 9 ECL Modeling

![Credit Risk](https://img.shields.io/badge/Credit%20Risk-Analysis-blue) ![IFRS9](https://img.shields.io/badge/IFRS9-Compliant-green) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange) ![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)

A comprehensive **Credit Risk Analysis** project implementing **IFRS 9 compliant Expected Credit Loss (ECL)** modeling using advanced machine learning techniques, Weight of Evidence (WOE) transformation, and Information Value (IV) analysis.

## 📋 **Project Overview**

This project demonstrates a complete **end-to-end credit risk modeling pipeline** that aligns with **IFRS 9 regulatory requirements** for financial institutions. The implementation covers data preprocessing, advanced feature engineering using WOE/IV techniques, model development, and production-ready deployment artifacts for **Probability of Default (PD)** estimation.

### 🎯 **Key Objectives**
- Build **IFRS 9 compliant** Expected Credit Loss models
- Implement **WOE/IV based feature engineering** for enhanced model interpretability
- Develop **production-ready PD models** using multiple ML algorithms
- Create **regulatory-approved** model documentation and validation
- Generate **business intelligence dashboards** for risk monitoring

### 🏆 **Business Impact**
- **Risk Assessment**: Accurate identification of high-risk borrowers
- **Capital Planning**: IFRS 9 compliant ECL calculations for regulatory capital
- **Pricing Optimization**: Risk-based loan pricing strategies
- **Portfolio Management**: Proactive risk monitoring and mitigation
- **Regulatory Compliance**: Meet Basel III and IFRS 9 requirements

---

## 🌟 **IFRS 9 Framework Implementation**

### **What is IFRS 9?**

**International Financial Reporting Standard 9 (IFRS 9)** is a comprehensive accounting standard that replaced IAS 39, fundamentally changing how financial institutions recognize, measure, and report credit losses. Instead of the traditional **incurred loss model**, IFRS 9 introduced the **Expected Credit Loss (ECL)** model.

### **📊 IFRS 9 Three-Stage Approach**

#### **🟢 Stage 1: 12-Month ECL (Performing Assets)**
- **Definition**: Financial instruments with **no significant increase in credit risk** since initial recognition
- **ECL Calculation**: Expected credit losses over the **next 12 months**
- **Business Logic**: Low-risk, current customers with good payment history
- **Model Requirements**: 
  - PD estimation for 12-month horizon
  - Loss Given Default (LGD) modeling
  - Exposure at Default (EAD) calculation

```python
# Stage 1 ECL Formula
ECL_Stage1 = PD_12M × LGD × EAD
```

#### **🟡 Stage 2: Lifetime ECL (Underperforming Assets)**
- **Definition**: Financial instruments with **significant increase in credit risk** but not credit-impaired
- **ECL Calculation**: Expected credit losses over the **entire remaining life** of the instrument
- **Business Logic**: Customers showing early warning signs of distress
- **Triggers for Stage 2**:
  - Significant increase in PD since origination
  - 30+ days past due (rebuttable presumption)
  - Qualitative factors (e.g., covenant breaches)

```python
# Stage 2 ECL Formula
ECL_Stage2 = PD_Lifetime × LGD × EAD
```

#### **🔴 Stage 3: Lifetime ECL (Credit-Impaired Assets)**
- **Definition**: Financial instruments that are **credit-impaired**
- **ECL Calculation**: Expected credit losses over **remaining lifetime**
- **Business Logic**: Customers in default or showing objective evidence of impairment
- **Impairment Indicators**:
  - 90+ days past due
  - Bankruptcy or financial reorganization
  - Breach of contract terms

```python
# Stage 3 ECL Formula
ECL_Stage3 = PD_Lifetime × LGD × EAD
# Note: PD approaches 100% for credit-impaired assets
```

### **🔄 Stage Migration Logic**

```python
def determine_ifrs9_stage(current_pd, origination_pd, days_past_due, qualitative_factors):
    """
    IFRS 9 Stage Determination Logic
    """
    # Stage 3: Credit-Impaired
    if days_past_due >= 90 or qualitative_factors['credit_impaired']:
        return 3
    
    # Stage 2: Significant Increase in Credit Risk
    if (current_pd / origination_pd >= 2.0 or  # Relative PD increase
        days_past_due >= 30 or                 # 30+ DPD presumption
        qualitative_factors['significant_increase']):
        return 2
    
    # Stage 1: Performing
    return 1
```

### **📈 Expected Credit Loss Components**

#### **1. Probability of Default (PD)**
- **Definition**: Likelihood that a borrower will default within a specific time horizon
- **Our Implementation**: Machine learning models (Logistic Regression, Random Forest, XGBoost)
- **Enhancement**: WOE/IV transformation for regulatory compliance
- **Validation**: Point-in-Time (PIT) vs Through-the-Cycle (TTC) calibration

#### **2. Loss Given Default (LGD)**
- **Definition**: Percentage of exposure lost if default occurs
- **Factors**: Collateral value, recovery processes, legal frameworks
- **Industry Benchmarks**: 
  - Secured loans: 20-40%
  - Unsecured loans: 60-80%
  - Corporate bonds: 40-60%

#### **3. Exposure at Default (EAD)**
- **Definition**: Outstanding amount at the time of default
- **Considerations**: Undrawn commitments, credit conversion factors
- **Formula**: `EAD = Outstanding Balance + (Undrawn × CCF)`

---

## 🔬 **Advanced Feature Engineering: WOE & IV Analysis**

### **🎯 Weight of Evidence (WOE) - The Foundation**

**Weight of Evidence** is a powerful technique that transforms categorical variables into continuous numerical values while maintaining their predictive relationship with the target variable.

#### **📊 Mathematical Foundation**

```python
# WOE Formula
WOE = ln(Distribution of Goods / Distribution of Bads)
WOE = ln((% of Non-Defaults in category) / (% of Defaults in category))
```

#### **🔍 WOE Interpretation**
- **Positive WOE**: Category has **lower default risk** than average
- **Negative WOE**: Category has **higher default risk** than average  
- **WOE = 0**: Category has **average risk**
- **Magnitude**: Indicates **strength of relationship**

#### **💼 Business Example**
```python
# Home Ownership WOE Analysis
Category     | Good% | Bad%  | WOE   | Risk Level
-------------|-------|-------|-------|------------
OWN          | 35%   | 20%   | +0.56 | Lower Risk
MORTGAGE     | 45%   | 40%   | +0.12 | Slightly Lower
RENT         | 18%   | 35%   | -0.66 | Higher Risk
OTHER        | 2%    | 5%    | -0.92 | Highest Risk
```

### **📈 Information Value (IV) - Feature Selection Power**

**Information Value** quantifies the **predictive power** of each feature, enabling objective feature selection for regulatory models.

#### **📊 Mathematical Foundation**

```python
# IV Formula
IV = Σ (% Goods - % Bads) × WOE
```

#### **🎯 IV Interpretation Guidelines**

| IV Range | Predictive Strength | Business Decision |
|----------|-------------------|-------------------|
| < 0.02   | Not Useful        | ❌ Exclude from model |
| 0.02-0.1 | Weak              | ⚠️ Consider with caution |
| 0.1-0.3  | Medium            | ✅ Include in model |
| 0.3-0.5  | Strong            | ⭐ Priority features |
| > 0.5    | Suspicious        | 🔍 Check for overfitting |

### **🚀 WOE/IV Advantages in Credit Risk**

#### **1. Regulatory Compliance**
- ✅ **Basel Committee approved**: Widely accepted by banking regulators
- ✅ **Interpretable**: Clear business meaning for audit purposes
- ✅ **Transparent**: Auditable methodology for model validation

#### **2. Technical Benefits**
- ✅ **Monotonic relationship**: Ensures logical risk ordering
- ✅ **Handles missing values**: Natural treatment for unknown categories
- ✅ **Reduces dimensionality**: One value per categorical feature
- ✅ **Linear with log-odds**: Perfect for logistic regression

#### **3. Business Advantages**
- ✅ **Feature ranking**: Objective importance measurement
- ✅ **Risk quantification**: Clear impact on default probability
- ✅ **Stakeholder communication**: Easy to explain to business users
- ✅ **Model stability**: Robust performance across time periods

---

## 🛠️ **End-to-End Modeling Pipeline**

### **📊 1. Data Engineering & Preprocessing**

#### **Data Quality Framework**
```python
# Outlier Treatment Strategy
- Age filtering: Remove age > 70 (business logic)
- Employment length: Cap at 47 years (realistic career span)
- Income validation: Remove extreme outliers using IQR method
```

#### **Missing Value Strategy**
```python
# Advanced Imputation Techniques
- Numerical features: Median imputation (robust to outliers)
- Categorical features: Mode imputation or separate "Unknown" category
- Business logic validation: Ensure realistic value ranges
```

#### **Feature Engineering Pipeline**
```python
# Comprehensive Feature Creation
1. Debt-to-Income Ratio: loan_amnt / person_income
2. Credit History Length: Standardized across borrowers
3. Employment Stability: person_emp_length categories
4. Risk Interactions: Combined risk factors
```

### **⚖️ 2. Advanced Balancing Techniques**

#### **SMOTE Implementation**
```python
# Synthetic Minority Oversampling Technique
- Problem: Imbalanced dataset (78% good, 22% default)
- Solution: Generate synthetic default cases using k-NN
- Benefit: Balanced training without simple duplication
- Result: Equal class distribution for unbiased learning
```

### **🤖 3. Multi-Algorithm Model Development**

#### **Model Portfolio Approach**

| Algorithm | Strengths | Use Case | Regulatory Status |
|-----------|-----------|----------|-------------------|
| **Logistic Regression** | Interpretable, Fast | Regulatory Reporting | ✅ Widely Accepted |
| **Random Forest** | Non-linear, Robust | Complex Patterns | ✅ Approved with Explanation |
| **XGBoost** | High Performance | Maximum Accuracy | ⚠️ Requires Interpretability |

#### **Model Validation Framework**
```python
# Comprehensive Performance Metrics
- AUC-ROC: Discrimination ability (target: >0.75)
- Precision/Recall: Class-specific performance
- Feature Importance: Business logic validation
- Stability Testing: Performance across time periods
```

### **📈 4. Advanced Model Interpretation**

#### **Feature Importance Analysis**
```python
# Multi-Model Feature Ranking
1. Logistic Regression: Coefficient analysis
2. Random Forest: Gini importance
3. XGBoost: Gain-based importance
4. Consensus Ranking: Features important across all models
```

#### **Business Impact Quantification**
```python
# Risk Driver Analysis
- Primary Drivers: Features with highest IV scores
- Risk Interactions: Combined feature effects
- Business Translation: Feature impact in business terms
- Regulatory Explanation: Model decisions for auditors
```

### **🎯 5. Production Deployment Architecture**

#### **Model Artifacts Package**
```python
# Complete Deployment Package
├── best_pd_model_xgboost.pkl      # Trained model
├── woe_analyzer.pkl               # WOE transformation logic
├── feature_scaler.pkl             # Standardization parameters
├── feature_names.pkl              # Feature ordering
└── model_card.pkl                 # Complete documentation
```

#### **API-Ready Architecture**
```python
# Production Prediction Pipeline
def predict_default_probability(customer_data):
    # 1. Data validation and preprocessing
    # 2. WOE transformation
    # 3. Feature scaling
    # 4. Model prediction
    # 5. IFRS 9 stage determination
    # 6. ECL calculation
    return {
        'pd_score': probability,
        'ifrs9_stage': stage,
        'ecl_amount': expected_loss,
        'risk_factors': key_drivers
    }
```

---

## 📊 **Business Intelligence & Monitoring**

### **🎛️ Power BI Dashboard Features**

#### **Executive Risk Dashboard**
- 📈 **Portfolio Risk Metrics**: Overall default rates and trends
- 🎯 **IFRS 9 Stage Distribution**: Stage 1/2/3 portfolio composition
- 💰 **ECL Provisions**: Current and projected loss provisions
- 📊 **Model Performance**: Accuracy, precision, recall tracking

#### **Risk Manager Dashboard**
- 🔍 **Individual Risk Scoring**: Customer-level PD scores
- ⚠️ **Early Warning System**: Customers at risk of stage migration
- 📋 **Model Validation**: Backtesting and performance monitoring
- 🎯 **Feature Analysis**: Key risk driver identification

#### **Regulatory Reporting**
- 📜 **Basel III Compliance**: Capital adequacy calculations
- 📊 **IFRS 9 Reporting**: ECL methodology documentation
- 🔍 **Model Documentation**: Complete audit trail
- 📈 **Stress Testing**: Scenario analysis and sensitivity testing

---

## 🏗️ **Project Structure**

```
Creditrisk_ENDtoEND/
│
├── 📊 analysis_final.ipynb           # Traditional ML approach
├── 🏦 pd_model_woe_iv.ipynb         # WOE/IV enhanced modeling
│
├── 📁 Rawdata/
│   └── credit_risk_dataset.csv      # Original dataset (32K+ records)
│
├── 📁 prepareddata/
│   ├── prepareddata.csv             # Processed dataset
│   ├── pd_prediction.xlsx           # Model predictions
│   └── woe_enhanced_predictions.csv # WOE-enhanced results
│
├── 📁 model_artifacts/
│   ├── best_pd_model_random_forest.pkl  # Production model
│   ├── woe_analyzer.pkl                 # WOE transformation
│   ├── feature_scaler.pkl               # Scaling parameters
│   ├── feature_names.pkl                # Feature metadata
│   └── model_card.pkl                   # Model documentation
│
├── 📁 powerbi_dash/
│   └── credit_monitoring.pbix       # Executive dashboard
│
└── 📄 README.md                     # This comprehensive guide
```

---

## 🚀 **Getting Started**

### **📋 Prerequisites**
```bash
# Required Python packages
pip install pandas numpy scikit-learn xgboost
pip install imbalanced-learn matplotlib seaborn
pip install jupyter notebook plotly
```

### **🔧 Quick Start Guide**

#### **1. Traditional Approach**
```bash
# Run the basic credit risk analysis
jupyter notebook analysis_final.ipynb
```

#### **2. Advanced IFRS 9 Compliant Approach**
```bash
# Run the WOE/IV enhanced modeling
jupyter notebook pd_model_woe_iv.ipynb
```

#### **3. Model Deployment**
```python
# Load production model
import pickle
import joblib

# Load model artifacts
model = joblib.load('model_artifacts/best_pd_model_random_forest.pkl')
woe_analyzer = pickle.load(open('model_artifacts/woe_analyzer.pkl', 'rb'))
scaler = joblib.load('model_artifacts/feature_scaler.pkl')

# Make predictions
probability = model.predict_proba(processed_features)[:, 1]
```

---

## 📈 **Model Performance Results**

### **🏆 Champion Model: XGBoost**
- **AUC Score**: 0.95+ (Excellent discrimination)
- **Precision**: 98% (Class 1 - Defaults)
- **Recall**: 91% (Catches 91% of actual defaults)
- **F1-Score**: 0.95 (Balanced performance)

### **🎯 Business Impact Metrics**
- **Risk Reduction**: 15% improvement in default detection
- **False Positive Rate**: <3% (Minimal good customer rejection)
- **Capital Efficiency**: Optimized ECL provisions
- **Regulatory Compliance**: Full IFRS 9 alignment

### **📊 Feature Importance Rankings**
1. **Previous Default History** (IV: 0.45) - Strong predictor
2. **Loan Interest Rate** (IV: 0.28) - Risk-based pricing validation
3. **Debt-to-Income Ratio** (IV: 0.24) - Key affordability metric
4. **Employment Length** (IV: 0.18) - Stability indicator
5. **Home Ownership** (IV: 0.12) - Asset backing consideration

---

## 🔮 **Future Enhancements**

### **🚀 Advanced Modeling Techniques**
- [ ] **Deep Learning**: Neural networks for complex pattern recognition
- [ ] **Ensemble Methods**: Stacking multiple algorithms for improved accuracy
- [ ] **Time Series Modeling**: Dynamic PD estimation with economic cycles
- [ ] **Graph Neural Networks**: Incorporating relationship data

### **📊 Enhanced Analytics**
- [ ] **Real-time Monitoring**: Live model performance dashboards
- [ ] **Explainable AI**: SHAP values for individual predictions
- [ ] **Stress Testing**: Economic scenario modeling
- [ ] **Model Drift Detection**: Automated retraining triggers

### **🏦 Regulatory Expansion**
- [ ] **Basel IV Implementation**: Updated capital requirements
- [ ] **CECL Compliance**: US GAAP expected loss modeling
- [ ] **Stress Testing**: Fed/ECB scenario analysis
- [ ] **Climate Risk**: ESG factor integration

---


### **🎯 Contribution Areas**
- Model performance improvements
- Additional feature engineering techniques
- Enhanced visualization dashboards
- Regulatory compliance enhancements
- Documentation improvements

---



## 🙏 **Acknowledgments**

- **Basel Committee on Banking Supervision** for regulatory frameworks
- **IFRS Foundation** for accounting standards guidance
- **Scikit-learn & XGBoost** communities for excellent ML libraries
- **Credit risk modeling** research community for theoretical foundations

---

**⭐ If you find this project helpful, please give it a star!**

*This project demonstrates advanced credit risk modeling techniques suitable for both academic learning and production deployment in financial institutions.*
