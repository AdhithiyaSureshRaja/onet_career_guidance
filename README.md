# O*NET Career Path Recommender

A comprehensive career guidance system that uses O*NET database data to provide personalized career recommendations based on user assessments of work values, interests, and work styles.

## 🎯 Project Overview

This project implements a machine learning-based career recommendation system that analyzes user responses to a 52-question assessment and matches them with occupations from the O*NET database using cosine similarity and multi-output regression.

## 🏗️ System Architecture

### Core Components
- **Data Processing Engine**: Processes O*NET database files (Work Values, Interests, Work Styles, Occupations)
- **Machine Learning Model**: Multi-output Random Forest Regressor for occupation profile prediction
- **Recommendation Engine**: Cosine similarity-based matching algorithm
- **Web Interface**: Streamlit-based user interface for assessment and results

### Data Sources
- **O*NET Database v29.3**: Comprehensive occupational information database
- **Work Values**: 6 dimensions (Achievement, Working Conditions, Relationships)
- **RIASEC Interests**: 6 personality types (Realistic, Investigative, Artistic, Social, Enterprising, Conventional)
- **Work Styles**: 16 behavioral dimensions (Leadership, Cooperation, Stress Tolerance, etc.)

## 📊 Features

### Assessment Dimensions
1. **Work Values (6 questions)**
   - Achievement & Recognition
   - Working Conditions & Support
   - Relationships & Independence

2. **RIASEC Interests (30 questions)**
   - Realistic: Working with things, machines, tools
   - Investigative: Working with ideas, research, analysis
   - Artistic: Working with creative expression
   - Social: Working with people, helping others
   - Enterprising: Working with people, leading, persuading
   - Conventional: Working with data, following procedures

3. **Work Styles (16 questions)**
   - Achievement & Persistence
   - Leadership & Social
   - Work Quality & Ethics
   - Stress & Adaptability

### Output
- Top 10 career recommendations with similarity scores
- CSV download functionality
- Real-time assessment scoring
- Normalized O*NET-compatible profiles

## 🚀 Installation & Setup

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn
```

### Data Files Required
- `db_29_3_text/` - O*NET database files
- `career_recommender_model.pkl` - Trained machine learning model

### Running the Application
```bash
streamlit run app.py
```

## 🔧 Model Training

The system uses a Jupyter notebook (`ONET_test.ipynb`) for training the recommendation model:

### Training Process
1. **Data Loading**: Loads O*NET text files and processes them into structured data
2. **Feature Engineering**: Creates occupation profiles from multiple O*NET dimensions
3. **Model Training**: Trains MultiOutputRegressor with RandomForestRegressor
4. **Model Persistence**: Saves trained model and feature data for the Streamlit app

### Model Performance
- **R² Score**: 0.9997
- **Features**: 22 normalized dimensions
- **Occupations**: 923 unique job titles
- **Model Size**: ~59 MB

## 📋 Assessment Questionnaire

The system uses a standardized 52-question assessment covering:

### Scoring System
- **Work Values**: 1-5 scale (Not Important to Extremely Important)
- **RIASEC Interests**: 1-5 scale (Strongly Dislike to Strongly Like)
- **Work Styles**: 1-5 scale (Strongly Disagree to Strongly Agree)

### Question Categories
- **Questions 1-6**: Work Values Assessment
- **Questions 7-36**: RIASEC Interest Assessment (5 questions per category)
- **Questions 37-52**: Work Styles Assessment

## 🎨 User Interface

### Streamlit App Features
- Clean, intuitive questionnaire interface
- Real-time response collection
- Instant career recommendations
- Downloadable results
- Responsive design with wide layout

### User Experience
- Slider-based input (1-5 scale)
- Session state management
- Progress tracking
- Immediate feedback and results

## 🔍 Recommendation Algorithm

### Matching Process
1. **Profile Creation**: Converts user responses to normalized O*NET profiles
2. **Model Prediction**: Uses trained model to predict ideal occupation profile
3. **Similarity Calculation**: Computes cosine similarity with all occupations
4. **Ranking**: Returns top matches based on similarity scores

### Technical Details
- **Normalization**: Scales user responses to 0-1 range matching O*NET scales
- **Feature Mapping**: Direct mapping to O*NET database dimensions
- **Similarity Metric**: Cosine similarity for profile comparison
- **Output**: Ranked list with similarity scores

## 📁 File Structure

```
onet_carrer_guidance/
├── app.py                          # Streamlit web application
├── ONET_test.ipynb                # Model training notebook
├── ONET_Questionnaire.txt         # Assessment questionnaire details
├── career_recommender_model.pkl   # Trained machine learning model
├── db_29_3_text/                 # O*NET database files
│   └── db_29_3_text/
│       ├── Work Values.txt
│       ├── Interests.txt
│       ├── Work Styles.txt
│       ├── Occupation Data.txt
│       └── [other O*NET files]
└── README.md                      # This file
```

## 🧪 Testing & Validation

### System Testing
- Sample neutral profile testing (all responses = 3)
- Model verification and loading tests
- Data integrity checks
- Feature column validation

### Quality Assurance
- Runtime warning handling
- Error handling for missing files
- Data type validation
- Column name detection

## 🔮 Future Enhancements

### Potential Improvements
- Additional assessment dimensions
- More sophisticated recommendation algorithms
- User profile saving and comparison
- Industry-specific filtering
- Educational pathway recommendations
- Salary and growth potential data

### Scalability
- Database integration for user management
- API endpoints for external applications
- Mobile app development
- Multi-language support

## 📚 Technical Specifications

### Machine Learning
- **Algorithm**: MultiOutputRegressor with RandomForestRegressor
- **Features**: 22 normalized O*NET dimensions
- **Training Data**: 923 occupation profiles
- **Validation**: 80/20 train-test split

### Data Processing
- **Input Format**: Tab-separated O*NET text files
- **Encoding**: UTF-8
- **Scaling**: Min-max normalization to 0-1 range
- **Missing Values**: Zero-filling strategy

### Performance
- **Model Training**: ~1-2 minutes
- **Inference Time**: <1 second per recommendation
- **Memory Usage**: ~60 MB for model + data
- **Scalability**: Handles 1000+ occupations efficiently

## 🤝 Contributing

This project is designed for career guidance and educational purposes. Contributions are welcome for:
- Model improvements
- UI/UX enhancements
- Additional assessment dimensions
- Documentation improvements

## 📄 License

This project uses O*NET data which is in the public domain. The machine learning model and application code are provided for educational and research purposes.

## 🆘 Support

For technical issues or questions about the career recommendation system, please refer to:
- O*NET database documentation
- Streamlit documentation
- Scikit-learn documentation

---

**Note**: This system provides career guidance based on O*NET data and should be used as one of many tools in career decision-making. Professional career counseling is recommended for major career decisions.
