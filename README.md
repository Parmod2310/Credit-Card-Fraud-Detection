<div align="center">
  <h1>Credit Card Fraud Detection</h1>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" alt="Project Status">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</div>

---

## 🚀 Overview
Credit card fraud is a significant challenge in the financial industry, causing financial losses and undermining customer trust. This project demonstrates a **machine learning-based approach** to detect fraudulent transactions using the Kaggle Credit Card Fraud Detection dataset.

## 🎯 Objectives
- **Detect fraudulent transactions:** Build a model to identify fraudulent activities with high accuracy.
- **Minimize false positives:** Reduce disruption to legitimate users by improving prediction precision.
- **Enhance security:** Provide a reliable and scalable solution for transaction safety.

## 📊 Dataset
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions with 31 features.
- **Key Features:**
  - `Time`: Time elapsed since the first transaction.
  - `V1-V28`: Principal components derived from PCA.
  - `Amount`: Transaction amount.
  - `Class`: Target variable (1 = Fraudulent, 0 = Legitimate).

---

## 🗂️ Project Structure
```
|-- Credit Card Fraud Detection.ipynb       # Jupyter Notebook with full implementation
|-- Credit Card Fraud Detection.pdf         # Detailed project report
|-- README.md                               # Project documentation
|-- fraud_detection_model_with_new_features.pkl # Trained Random Forest model with new features
|-- requirements.txt                        # Required Python libraries
```

---

## 🔑 Key Steps
1. **Data Preprocessing:**
   - Verified no missing values and removed duplicates.
   - Scaled numerical features using `StandardScaler`.
   - Handled class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

2. **Exploratory Data Analysis (EDA):**
   - Visualized class distribution.
   - Analyzed correlation matrix and feature importance.

3. **Dimensionality Reduction:**
   - Applied PCA to reduce features to 10 principal components, capturing 95% variance.

4. **Model Building:**
   - Models used: Random Forest, Logistic Regression, and XGBoost.
   - Employed cross-validation to ensure generalization.
   - Tuned hyperparameters using `GridSearchCV` and `RandomizedSearchCV`.

5. **Evaluation Metrics:**
   - **Accuracy**: Overall prediction correctness.
   - **Precision**: Correct positive predictions.
   - **Recall**: Sensitivity to fraud detection.
   - **F1 Score**: Balance between precision and recall.
   - **ROC-AUC**: Discrimination ability of the model.

---

## 📈 Results
| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Random Forest      | 99.94%   | 0.83      | 0.82   | 0.82     | 0.91    |
| Logistic Regression| 98.00%   | 0.07      | 0.90   | 0.12     | 0.53    |
| XGBoost            | 99.00%   | 0.13      | 0.84   | 0.22     | 0.56    |

The **Random Forest model** outperformed other models with the best balance between accuracy, precision, and recall.

---

## 🛠️ Dependencies
Install all required libraries using:
```bash
pip install -r requirements.txt
```

---

## 📖 Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Parmod2310/Credit-Card-Fraud-Detection.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook "Credit Card Fraud Detection.ipynb"
   ```
4. Explore the project report in `Credit Card Fraud Detection.pdf` for detailed insights.

---

## 🔮 Future Work
- **Advanced Models:** Explore LightGBM or deep learning for enhanced performance.
- **Real-Time Deployment:** Integrate the model into live transaction monitoring systems.
- **Feature Engineering:** Develop domain-specific features to improve model accuracy.

---

## 🙏 Acknowledgments
- Kaggle for providing the dataset.
- Open-source libraries (Pandas, Scikit-learn, XGBoost, Matplotlib, etc.) for enabling this project.

---

## 📬 Contact
For questions or collaboration, reach out via:
- **LinkedIn:** [LinkedIn](https://www.linkedin.com/in/parmod2310/)
- **GitHub:** [GitHub](https://github.com/Parmod2310)
- **Email:** p921035@gmail.com
