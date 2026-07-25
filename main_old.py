# server code
import os
import pickle
from typing import Optional, Tuple, List, Dict, Any

import numpy as np 
import pandas as pd
import app
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY not found in environment variables. Please set it in your .env file.")


app = FastAPI(title="Movie Recommendation System API", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #for local streamlit
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

#Global variables to hold the data and models


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

# Load the data and models at startup

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

# using pydantic

class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    vote_average: Optional[float] = None

class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview:Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []   #jonra

class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TFIDFRecItem]

def _norm_title(t: str) -> str:
    return str(t).strip().lower()



def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


#search functions
async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:

    """
    safe TMDB GET:
    -Network errorsm -> 502
    -TMDB API errors -> 502 with detail
    """

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )
    
    if r.status_code !=200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error (r.status_code): {r.text}"
        )
    
    return r.json()


async def tmdb_cards_from_results(
        results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    """
    Convert TMDB search results to TMDBMovieCard list
    """
    cards = []
    for r in results[:limit]:
        card = TMDBMovieCard(
            tmdb_id=r.get("id"),
            title=r.get("title"),
            release_date=r.get("release_date"),
            poster_url=make_img_url(r.get("poster_path")),
            vote_average=r.get("vote_average"),
        )
        cards.append(card)
    return cards