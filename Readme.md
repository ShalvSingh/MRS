# MRS — Movie Recommendation System

A personal movie recommendation system built for learning and portfolio purposes. It uses TMDB movie data to fetch movie details, posters, ratings, genres, and recommendations.

## Live Demo

- Frontend: add your Streamlit link here
- Backend/API: add your Render link here

## Features

- Search movies and get similar recommendations
- Fetch movie posters, ratings, genres, and overviews from TMDB
- FastAPI backend for recommendation logic
- Streamlit frontend for an easy interactive UI
- Saved model artifacts for fast inference

## Tech Stack

- Python
- FastAPI
- Streamlit
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Pickle
- TMDB API
- Render
- GitHub

## How It Works

The project uses content-based filtering.

1. Movie data is loaded from the dataset.
2. Text features are converted into vectors using TF-IDF.
3. Cosine similarity is used to compare movies.
4. The most similar movies are returned as recommendations.

## Project Structure

```text
MRS/
├── app.py
├── main.py
├── requirements.txt
├── Readme.md
├── df.pkl
├── indices.pkl
├── tfidf.pkl
├── tfidf_matrix.pkl
├── LICENSE
├── Movies.ipynb
├── IMAGES/
└── .devcontainer/



Installation

Clone the repository:

git clone https://github.com/ShalvSingh/MRS.git
cd MRS

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

Install dependencies:

pip install -r requirements.txt
Run Locally
Start the backend
uvicorn app:app --reload
Start the Streamlit app
streamlit run main.py
Requirements

Make sure your Python version matches the one used for deployment, recommended:

Python 3.11
Screenshots

Add screenshots here of:

Home page
Search result page
Recommendation output
API response

Example:

![Home](IMAGES/home.png)
![Recommendations](IMAGES/recommendations.png)
Future Improvements
Add user login and watchlist
Add collaborative filtering
Add hybrid recommendations
Add Docker support
Add CI/CD pipeline
Improve UI and mobile responsiveness
License

This project is licensed under the MIT License.

Author
Aman Singh
GitHub: ShalvSingh