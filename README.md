# SegmentIQ
### AI-Powered Customer Segmentation using Machine Learning

<img width="1439" height="748" alt="Screenshot 2026-06-27 at 1 42 41 PM" src="https://github.com/user-attachments/assets/ed1a3d85-0a4b-4023-923f-eb3dd5d3fe00" />


SegmentIQ is an interactive machine learning application that groups customers into meaningful behavioral segments using **K-Means Clustering**. The project helps businesses understand customer purchasing patterns and enables data-driven marketing strategies.

---

## Features

- Customer segmentation using K-Means clustering
- Interactive dashboard built with Streamlit
- Real-time cluster prediction
- Customer profile analysis
- Feature contribution visualization
- Business-friendly cluster descriptions
- Clean dark-themed UI

---

## Dashboard

The application allows users to enter customer attributes such as:

- Age
- Annual Income
- Total Spending
- Web Purchases
- Store Purchases
- Web Visits
- Recency

The model predicts the customer's segment and displays useful business insights.

---

## Machine Learning Pipeline

```
Customer Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Scaling
      │
      ▼
K-Means Clustering
      │
      ▼
Cluster Prediction
      │
      ▼
Interactive Dashboard
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| ML | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Frontend | Streamlit |
| Styling | Custom CSS |

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/SegmentIQ.git
```

Move into the project directory

```bash
cd SegmentIQ
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Customer Segments

| Cluster | Description |
|----------|-------------|
| 0 | High Value Customers |
| 1 | Budget Buyers |
| 2 | Frequent Shoppers |
| 3 | Dormant High Spenders |
| 4 | Loyal Customers |

*(Descriptions may vary depending on the trained model.)*

---

## Future Improvements

- Automatic retraining
- Support for multiple clustering algorithms
- SHAP feature explanations
- Customer recommendation engine
- Export segmentation reports
- Cloud deployment

---

## Screenshots

<img src="assets/dashboard.png" width="100%">

---

## Author

**Himanish Rana**

Electrical & Computer Engineering Student

Interested in AI, Machine Learning, LLMs, FPGA Accelerators and Data Science.

---

## License

MIT License
