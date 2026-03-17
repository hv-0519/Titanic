# Titanic Survival Prediction 🚢

A full-stack machine learning application that predicts passenger survival outcomes from the Titanic dataset using multiple classification algorithms. This project features a Flask backend API, an interactive web frontend, and three trained ML models for comparison.

## 📋 Project Overview

This application demonstrates end-to-end ML deployment by training three different classification models on the famous Titanic dataset and providing:
- **Real-time predictions** for survival likelihood
- **Model comparison** with performance metrics
- **Interactive visualizations** including ROC curves, confusion matrices, and feature importance
- **RESTful API endpoints** for programmatic access
- **Production-ready deployment** on Render

## 🎯 Features

### Machine Learning Models
- **Logistic Regression** - Fast, interpretable baseline model
- **K-Nearest Neighbors (KNN)** - Instance-based learning with k-parameter optimization
- **Decision Tree** - Non-linear classification with feature importance analysis

### Web Interface
- Interactive prediction form with real-time results
- Model performance dashboard with accuracy metrics
- ROC curve visualization for model evaluation
- Confusion matrix visualization
- KNN k-parameter optimization chart
- Feature importance analysis

### API Endpoints
- `POST /predict` - Get survival prediction for passenger
- `GET /model-metrics` - Retrieve model performance metrics
- `GET /roc-data/<model_name>` - Get ROC curve data
- `GET /confusion-matrix/<model_name>` - Get confusion matrix
- `GET /knn-k-scores` - Get KNN k-parameter scores
- `GET /feature-importance` - Get feature importance values
- `GET /features` - List available features

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/hv-0519/Titanic.git
cd Titanic
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

The app will start on `http://localhost:5000`

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.0.0 | Web framework |
| Flask-Cors | 4.0.0 | CORS support |
| scikit-learn | Latest | ML algorithms & metrics |
| pandas | Latest | Data manipulation |
| numpy | Latest | Numerical computing |
| seaborn | Latest | Titanic dataset & visualization |
| matplotlib | Latest | Data visualization |
| gunicorn | 25.1.0 | Production server |

See `requirements.txt` for complete list.

## 🚀 Deployment

### Deploy on Render

1. Push code to GitHub
2. Connect repository to Render
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`
5. Deploy!

The application uses the `Procfile` and `render.yaml` configuration files for deployment.

## 📊 Dataset & Preprocessing

- **Source**: Titanic dataset (via seaborn library)
- **Records**: ~891 passengers
- **Target**: Binary classification (Survived/Did Not Survive)

### Data Cleaning Pipeline
- ✅ Handles missing values (age, fare, embarked)
- ✅ Encodes categorical features (sex, embarked)
- ✅ Removes irrelevant columns
- ✅ Validates no NaN values remain

### Features Used
- pclass (passenger class)
- sex
- age
- sibsp (siblings/spouses aboard)
- parch (parents/children aboard)
- fare
- embarked (port of embarkation)
- alone (traveling alone)

## 📈 Model Performance

Models are pre-trained on 80% training set (20% test split) with metrics computed including:
- Train & Test Accuracy
- ROC-AUC Score
- Confusion Matrix
- Feature Importances (for Decision Tree)

Performance metrics are cached and served via API for instant dashboard loading.

## 🔧 API Usage Examples

### Get a Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [3, 1, 22.0, 1, 0, 7.25, 2, 1],
    "model_name": "logistic"
  }'
```

### Get Model Metrics
```bash
curl http://localhost:5000/model-metrics
```

### Get ROC Data
```bash
curl http://localhost:5000/roc-data/logistic
```

## 📂 Project Structure

```
Titanic/
├── app.py              # Flask backend & ML models
├── index.html          # Interactive web interface
├── requirements.txt    # Python dependencies
├── Procfile           # Heroku/Render deployment config
├── render.yaml        # Render-specific configuration
└── README.md          # This file
```

## 🎓 Learning Resources

This project demonstrates:
- Data preprocessing & feature engineering
- Training & evaluation of multiple ML models
- Building REST APIs with Flask
- Frontend-backend integration
- Production deployment
- Model comparison & interpretation

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**hv-0519**

## 🔗 Live Demo

Visit the deployed application on Render:https://titanic-2-p866.onrender.com

## ❓ FAQ

**Q: How accurate are the predictions?**
A: Test accuracies range from 75-82% depending on the model. Use the dashboard to compare performance.

**Q: Can I use this for real predictions?**
A: This is for educational purposes. The Titanic dataset is historical and the model is trained only on that specific dataset.

**Q: Why multiple models?**
A: To demonstrate model comparison and showcase different algorithm strengths for classification tasks.

**Q: How long does prediction take?**
A: Typically <100ms for inference. All models are pre-trained at startup.

## 🐛 Troubleshooting

- **Port already in use**: Change port with `python app.py --port 8000`
- **Missing dependencies**: Run `pip install -r requirements.txt --upgrade`
- **CORS errors**: Flask-CORS is configured to allow all origins
- **Deployment timeout**: Check `Procfile` timeout setting (120s)
