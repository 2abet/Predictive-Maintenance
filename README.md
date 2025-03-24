# Predictive Maintenance Dashboard – AI vs Threshold

This Streamlit app demonstrates a **side-by-side comparison of traditional rule-based alerts vs AI-powered predictions** for predictive maintenance. It's tailored for industrial environments like chocolate manufacturing, where minimizing machine downtime is critical.

---

## 🚀 Features

- Simulates sensor data (vibration, temperature, current)
- Injects borderline failure cases for realistic scenarios
- Compares:
  - Rule-based threshold alerts (manual)
  - AI model (Random Forest) predictions
- Live sliders to test failure risk
- Confusion matrix comparison for performance insight

---

## 🧪 Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- scikit-learn

---

## 📦 Installation

Clone the repo or download the ZIP.

```bash
pip install -r requirements.txt
streamlit run predictive_dashboard.py
```

---

## 💡 Why AI?

While thresholds work well in simple cases, they:
- Miss combined sensor patterns
- Trigger false alarms
- Require manual tuning

AI learns subtle relationships and adapts over time for earlier, smarter predictions.

---

## 📁 Files

- `predictive_dashboard.py` – Main app
- `requirements.txt` – Dependencies

---

## 🧠 Author

Akinyemi Arabambi

Demo built for showcasing AI potential in predictive maintenance at **Magna Confectionery**.

