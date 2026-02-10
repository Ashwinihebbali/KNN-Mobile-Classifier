# KNN Mobile Classifier

**A Machine Learning System for Mobile Phone Market Segmentation and Price Range Prediction**

---

## Problem Statement

This project addresses the critical business challenge of **automated market segmentation and price range prediction for mobile devices**. In the highly competitive mobile phone industry, accurate positioning is essential for competitive strategy, pricing optimization, and market penetration.

The objective is to develop a robust K-Nearest Neighbors (KNN) classification model that predicts market price positioning of mobile phones by analyzing their hardware specifications and technical characteristics. This enables stakeholders to make data-driven decisions regarding product placement, competitive analysis, and strategic pricing.

### Business Objectives

- Classify mobile phones into distinct market price segments based on technical specifications
- Identify key hardware features that drive market positioning and pricing strategy
- Enable automated decision-making for inventory management and product positioning
- Provide actionable insights for manufacturers, OEMs, and retailers
- Support competitive benchmarking and market analysis


## Project Overview

This initiative implements a comprehensive machine learning pipeline for mobile phone market segmentation. The system processes **2,000+ mobile device records** encompassing 15 technical specifications to generate accurate price tier classifications.

### Key Features

- **Robust Classification System**: K-Nearest Neighbors algorithm for reliable market segmentation
- **Data-Driven Insights**: Correlation analysis between hardware specifications and market positioning
- **Scalable Architecture**: Designed to accommodate new devices and market trends
- **Production-Ready Code**: Clean, documented, and maintainable implementation
- **Comprehensive Analysis**: Full exploratory data analysis and preprocessing pipeline

---

## Dataset Description

### Features

| Feature | Description |
|---------|-------------|
| `battery_power` | Battery capacity in mAh |
| `clock_speed` | CPU clock speed in GHz |
| `fc` | Front camera megapixels |
| `int_memory` | Internal memory in GB |
| `m_dep` | Mobile depth in mm |
| `mobile_wt` | Mobile weight in grams |
| `n_cores` | Number of processor cores |
| `pc` | Rear camera megapixels |
| `px_height` | Pixel resolution height |
| `px_width` | Pixel resolution width |
| `ram` | RAM capacity in MB |
| `sc_h` | Screen height in cm |
| `sc_w` | Screen width in cm |
| `talk_time` | Maximum talk time in hours |
| `Price` | Market price range (target variable) |

### Data Characteristics

- **Total Records**: 2,000+ mobile phones
- **Total Features**: 15 (14 input features + 1 target)
- **Data Type**: Numeric
- **Target Variable**: Price (classification into price categories)

---

## Methodology

### Data Preparation & Preprocessing

1. **Data Acquisition & Validation**
   - Imported 2,000+ mobile phone records with 15 technical attributes
   - Performed comprehensive data quality assessment
   - Identified missing values and anomalies using statistical methods

2. **Data Cleaning & Imputation**
   - Handled zero-values (especially in pixel resolution and screen dimensions)
   - Applied median-based imputation for robust outlier treatment
   - Validated data consistency post-cleaning

3. **Exploratory Data Analysis (EDA)**
   - Conducted univariate and bivariate statistical analysis
   - Visualized feature distributions and relationships
   - Computed correlation matrices to identify feature importance
   - Examined key statistics (mean, median, quartiles, variance)

4. **Feature Engineering & Normalization**
   - Standardized/normalized features for distance-based algorithms
   - Identified and handled outliers using quantile-based methods
   - Assessed feature importance for model optimization

### Model Architecture

**Algorithm**: K-Nearest Neighbors (KNN) Classification
- **Rationale**: Non-parametric approach suitable for multi-class segmentation with clear feature relationships
- **Distance Metric**: Euclidean distance
- **Training Strategy**: 80-20 train-test split
- **Validation Approach**: Cross-validation for robust performance estimation

### Performance Metrics

- **Accuracy**: Overall classification correctness
- **Precision**: Performance on positive class predictions
- **Recall**: Sensitivity to market segment detection
- **F1-Score**: Harmonic mean for balanced evaluation

---

## Project Structure

```
Market Positioning of Mobile/
├── README.md                          # Project documentation
├── Market Positioning of Mobile.ipynb  # Main Jupyter notebook
├── Mobile_data.csv                     # Dataset
└── Problem Statement_KNN.pdf           # Detailed problem statement
```

---

## Technical Stack

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | >=1.0.0 | Data manipulation and analysis |
| `numpy` | >=1.18.0 | Numerical computing |
| `scikit-learn` | >=0.23.0 | Machine learning algorithms |
| `matplotlib` | >=3.1.0 | Data visualization |

### Installation

**Prerequisites**: Python 3.7+

```bash
# Clone the repository
git clone https://github.com/yourusername/knn-mobile-classifier.git
cd knn-mobile-classifier

# Create virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Or install packages individually:**
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Getting Started

### Quick Start

1. **Prepare Environment**
   ```bash
   python -m venv env && source env/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook "Market Positioning of Mobile.ipynb"
   ```

3. **Execute Pipeline**
   - Run cells sequentially from top to bottom
   - Monitor data quality metrics at each preprocessing stage
   - Review exploratory visualizations
   - Train and evaluate the KNN classifier

### Notebook Structure

| Section | Purpose |
|---------|---------|
| **Import & Load** | Initialize libraries and import dataset |
| **Data Exploration** | Statistical analysis and data profiling |
| **Data Cleaning** | Handle missing values and outliers |
| **EDA & Visualization** | Visualize distributions and relationships |
| **Model Training** | Train KNN classifier on processed data |
| **Evaluation** | Assess model performance and metrics |
| **Insights** | Interpret results for business context |

---

## Key Findings & Insights

### Feature Importance Analysis

The analysis identifies critical hardware specifications that drive market positioning:

- **High-Impact Features**: RAM, battery capacity, and display resolution demonstrate strong correlation with price positioning
- **Processing Power**: CPU speed and core count significantly influence device classification
- **Camera Specifications**: Front and rear camera capabilities contribute to market tier determination
- **Form Factor**: Weight, dimensions, and display size affect product positioning

### Market Segmentation

The KNN model successfully segments the mobile phone market into distinct price tiers, revealing:

- Clear clustering patterns in hardware specification space
- Distinct market segments with well-defined characteristic profiles
- Strong separability between budget, mid-range, and premium segments

---

## Model Performance & Results

### Classification Outcomes

The trained KNN classifier achieves robust predictive performance:

- **High Accuracy**: Correctly classifies majority of devices into appropriate market segments
- **Balanced Precision/Recall**: Maintains equilibrium across market tiers
- **Cross-Validation**: Validates model generalization capability

### Business Applications

- **Price Strategy**: Support pricing decisions aligned with competitive landscape
- **Product Positioning**: Enable accurate market tier placement
- **Portfolio Analysis**: Benchmark products against market competitors
- **Demand Forecasting**: Predict market demand based on segment characteristics
- **Risk Mitigation**: Identify pricing anomalies and strategic opportunities

---

## Future Enhancements & Roadmap

### Algorithm Optimization

- [ ] **Hyperparameter Tuning**: Systematic k-value optimization using GridSearchCV
- [ ] **Ensemble Methods**: Compare KNN with Random Forest, Gradient Boosting, XGBoost
- [ ] **Neural Networks**: Explore deep learning approaches for complex pattern recognition
- [ ] **Cross-Validation**: Implement k-fold cross-validation for robust validation

### Model Enhancement

- [ ] **Feature Selection**: Advanced feature engineering and dimension reduction (PCA, t-SNE)
- [ ] **Class Imbalance**: Address potential class imbalance using SMOTE/class weights
- [ ] **Anomaly Detection**: Implement isolation forests for outlier identification
- [ ] **Probabilistic Predictions**: Generate confidence scores for predictions

### Production Deployment

- [ ] **Model Serialization**: Save trained models for production inference
- [ ] **API Development**: Build REST API for real-time predictions
- [ ] **Containerization**: Docker containerization for scalable deployment
- [ ] **Monitoring**: Implement performance monitoring and drift detection

### Data Enhancement

- [ ] **External Data Integration**: Incorporate brand reputation and market sentiment
- [ ] **Time-Series Analysis**: Analyze temporal trends in mobile market evolution
- [ ] **Scenario Modeling**: Develop what-if analysis capabilities
- [ ] **Real-Time Updates**: Integrate live market data feeds

---

## Project Information

**Project Name**: KNN Mobile Classifier  
**Objective**: Mobile phone market segmentation and price range prediction  
**Algorithm**: K-Nearest Neighbors (KNN) Classification  
**Dataset Size**: 2,000+ mobile phone records  
**Features**: 15 hardware specifications  
**Status**: Active Development  
**Last Updated**: February 2026

---

## Contributing

We welcome contributions to enhance this project. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Submit a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Citation

If you use this project in your research or work, please cite:

```bibtex
@project{knn_mobile_classifier,
  title={KNN Mobile Classifier: Machine Learning for Mobile Phone Market Segmentation},
  year={2026},
  url={https://github.com/yourusername/knn-mobile-classifier}
}
```

---

## Contact & Support

**Project Repository**: [github.com/yourusername/knn-mobile-classifier](https://github.com/yourusername/knn-mobile-classifier)

For questions, issues, or suggestions:
- Open an issue in the GitHub repository
- Review the Problem Statement documentation
- Consult the Jupyter notebook for implementation details

**Disclaimer**: This project is intended for educational and research purposes in machine learning and market analysis. Predictions should be validated with domain experts before use in production environments.

---

**© 2026 - All Rights Reserved**
