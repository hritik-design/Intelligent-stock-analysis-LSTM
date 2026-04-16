# 📈 Intelligent Stock Market Analysis & Prediction using LSTM

## 🚀 Project Description

An interactive web application built using Streamlit that performs stock market analysis and predicts future stock prices using a Long Short-Term Memory (LSTM) deep learning model. The system provides visual insights such as moving averages, daily returns, correlation heatmaps, and risk vs return analysis.

---

## 📌 Key Features

* 📊 Real-time stock data using Yahoo Finance API
* 📉 Moving averages (10, 20, 50 days)
* 📈 Daily returns and distribution analysis
* 🔗 Correlation heatmaps between multiple stocks
* ⚠️ Risk vs Return visualization
* 🤖 LSTM-based stock price prediction
* 🎯 Interactive dashboard using Streamlit

---

## 🧠 Technologies Used

* Python
* Streamlit
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* TensorFlow / Keras
* yFinance API

---

## 📂 Project Structure

```
├── app.py
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/intelligent-stock-analysis-lstm.git
cd intelligent-stock-analysis-lstm
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```
streamlit run app.py
```

---

## 📊 How It Works

* Fetches historical stock data using yFinance
* Performs exploratory data analysis (EDA)
* Calculates indicators like moving averages and returns
* Uses LSTM model trained on historical closing prices
* Predicts future stock prices based on past trends

---

## 📷 Output Screens

* Stock price trends
* Volume analysis
* Correlation heatmaps
* Risk vs return scatter plot
* LSTM prediction graph

---

## 📈 Model Details

* Model Type: LSTM (Deep Learning)
* Input: 60-day historical stock prices
* Training Split: 95%
* Evaluation Metric: RMSE (Root Mean Squared Error)

---

## ⚠️ Disclaimer

This project is for educational purposes only. It does not provide financial advice or guarantee accurate stock predictions.

---

## 👨‍💻 Author

Hritik Raj

---

## ⭐ GitHub Description (Short)

AI-powered stock market analysis and prediction system using LSTM with an interactive Streamlit dashboard.
