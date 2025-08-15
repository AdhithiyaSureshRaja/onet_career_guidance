# Deployment Guide for Streamlit Cloud

## Prerequisites
- A GitHub account
- Your code pushed to a GitHub repository
- A Streamlit Cloud account (free tier available)

## Steps to Deploy

### 1. Push Your Code to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to: `app.py`
6. Click "Deploy!"

## Important Notes

### Model File
- Make sure `career_recommender_model.pkl` is included in your repository
- This file should be under 100MB for free tier
- If larger, consider using Git LFS or hosting the model elsewhere

### Dependencies
- `requirements.txt` is already configured with all necessary packages
- `packages.txt` includes system dependencies
- `.streamlit/config.toml` optimizes the app for deployment

### Troubleshooting
If you get the "inotify instance limit reached" error:
- This is a known issue with some Linux environments
- The config.toml file should help mitigate this
- Try redeploying if the error persists

## Local Testing
Before deploying, test locally:
```bash
streamlit run app.py
```

## File Structure for Deployment
```
onet_career_guidance/
├── app.py                          # Main Streamlit app
├── career_recommender_model.pkl    # Your trained model
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── .gitignore                     # Git ignore file
└── DEPLOYMENT.md                  # This file
```
