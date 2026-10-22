import os
import asyncio
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()
WATCHMODE_API_KEY = os.getenv("WATCHMODE_API_KEY")

WATCHMODE_BASE = "https://api.watchmode.com/v1"
# Watchmode returns fully-qualified poster/image URLs already (unlike TMDB,
# which returns a path you have to prefix), so no image-base-url helper is needed.

if not WATCHMODE_API_KEY:
    raise RuntimeError("WATCHMODE_API_KEY missing. Put it in .env as WATCHMODE_API_KEY=xxxx")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API (Watchmode)", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class WatchmodeMovieCard(BaseModel):
    watchmode_id: int
    title: str
    poster_url: Optional[str] = None
    year: Optional[int] = None
    release_date: Optional[str] = None
    user_rating: Optional[float] = None


class WatchmodeSource(BaseModel):
    source_id: int
    name: str
    type: Optional[str] = None  # sub, rent, buy, free, tv_everywhere
    region: Optional[str] = None
    web_url: Optional[str] = None
    ios_url: Optional[str] = None
    android_url: Optional[str] = None
    price: Optional[float] = None
    format: Optional[str] = None


class WatchmodeMovieDetails(BaseModel):
    watchmode_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    year: Optional[int] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[str] = []  # genre names
    genre_ids: List[int] = []
    user_rating: Optional[float] = None
    critic_score: Optional[float] = None
    imdb_id: Optional[str] = None
    tmdb_id: Optional[int] = None
    trailer: Optional[str] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    watchmode: Optional[WatchmodeMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: WatchmodeMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[WatchmodeMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


async def watchmode_get(path: str, params: Dict[str, Any]) -> Any:
    """
    Safe Watchmode GET:
    - Network errors -> 502
    - Watchmode API errors -> 502 with detail
    """
    q = dict(params)
    q["apiKey"] = WATCHMODE_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{WATCHMODE_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Watchmode request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Watchmode error {r.status_code}: {r.text}"
        )

    return r.json()


def _year_to_release_date(year: Optional[int]) -> Optional[str]:
    return f"{year}-01-01" if year else None


async def watchmode_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[WatchmodeMovieCard]:
    """
    Normalizes both /list-titles/ items and /search/ title_results items,
    which use slightly different keys ('title' vs 'name', 'image_url', 'poster').
    """
    out: List[WatchmodeMovieCard] = []
    for m in (results or [])[:limit]:
        wid = m.get("id")
        if wid is None:
            continue
        out.append(
            WatchmodeMovieCard(
                watchmode_id=int(wid),
                title=m.get("title") or m.get("name") or "",
                poster_url=m.get("image_url") or m.get("poster"),
                year=m.get("year"),
                release_date=m.get("release_date") or _year_to_release_date(m.get("year")),
                user_rating=m.get("user_rating"),
            )
        )
    return out


async def watchmode_movie_details(watchmode_id: int) -> WatchmodeMovieDetails:
    data = await watchmode_get(f"/title/{watchmode_id}/details/", {})
    return WatchmodeMovieDetails(
        watchmode_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("plot_overview"),
        release_date=data.get("release_date") or _year_to_release_date(data.get("year")),
        year=data.get("year"),
        poster_url=data.get("poster"),
        backdrop_url=data.get("backdrop"),
        genres=data.get("genre_names", []) or [],
        genre_ids=data.get("genres", []) or [],
        user_rating=data.get("user_rating"),
        critic_score=data.get("critic_score"),
        imdb_id=data.get("imdb_id"),
        tmdb_id=data.get("tmdb_id"),
        trailer=data.get("trailer"),
    )


async def enrich_cards_with_posters(
    cards: List[WatchmodeMovieCard], concurrency: int = 8
) -> List[WatchmodeMovieCard]:
    """
    /list-titles/ and /search/ (used by /home, /recommend/genre, and
    /watchmode/search) do NOT reliably return poster images or ratings —
    only /title/{id}/details/ has real 'poster' and 'user_rating' fields.

    This fills in missing posters (and ratings, same call, no extra cost)
    by fetching details concurrently, capped by a semaphore so we don't
    blow through rate limits / quota. Each fetch costs 1 extra Watchmode
    credit, so this is opt-out via include_posters=False on the routes
    that use it.
    """
    sem = asyncio.Semaphore(concurrency)

    async def fetch_one(card: WatchmodeMovieCard) -> None:
        if card.poster_url and card.user_rating is not None:
            return
        async with sem:
            try:
                data = await watchmode_get(f"/title/{card.watchmode_id}/details/", {})
                if not card.poster_url:
                    card.poster_url = data.get("poster")
                if card.user_rating is None:
                    card.user_rating = data.get("user_rating")
            except Exception:
                pass  # leave fields as-is; card still renders with a placeholder

    await asyncio.gather(*(fetch_one(c) for c in cards))
    return cards


async def watchmode_title_sources(watchmode_id: int, region: str = "US") -> List[WatchmodeSource]:
    data = await watchmode_get(f"/title/{watchmode_id}/sources/", {"regions": region})
    out: List[WatchmodeSource] = []
    for s in data or []:
        out.append(
            WatchmodeSource(
                source_id=int(s.get("source_id")),
                name=s.get("name") or "",
                type=s.get("type"),
                region=s.get("region"),
                web_url=s.get("web_url"),
                ios_url=s.get("ios_url"),
                android_url=s.get("android_url"),
                price=s.get("price"),
                format=s.get("format"),
            )
        )
    return out


async def watchmode_search_movies(query: str, movies_only: bool = True) -> Dict[str, Any]:
    """
    Raw-ish Watchmode response for keyword search (MULTIPLE results).
    Streamlit can use this for suggestions and grid, similar to the old
    TMDB /search/movie passthrough.
    """
    data = await watchmode_get(
        "/search/",
        {"search_field": "name", "search_value": query},
    )
    if movies_only:
        results = data.get("title_results", []) or []
        data = {**data, "title_results": [r for r in results if r.get("type") == "movie"]}
    return data


async def watchmode_search_first(query: str) -> Optional[dict]:
    data = await watchmode_search_movies(query=query, movies_only=True)
    results = data.get("title_results", [])
    return results[0] if results else None


# =========================
# TF-IDF Helpers (unchanged — dataset-agnostic)
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_watchmode_card_by_title(title: str) -> Optional[WatchmodeMovieCard]:
    """
    Uses Watchmode search by title to fetch a poster/card for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await watchmode_search_first(title)
        if not m:
            return None
        card = WatchmodeMovieCard(
            watchmode_id=int(m["id"]),
            title=m.get("name") or title,
            poster_url=m.get("image_url"),
            year=m.get("year"),
            release_date=_year_to_release_date(m.get("year")),
        )
        if not card.poster_url:
            card = (await enrich_cards_with_posters([card]))[0]
        return card
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (Watchmode) ----------
@app.get("/home", response_model=List[WatchmodeMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
    include_posters: bool = Query(
        True, description="Fetch posters via extra detail calls (costs 1 credit/title)."
    ),
):
    """
    Home feed for Streamlit (posters), built on /list-titles/.

    Watchmode has no literal "trending" feed like TMDB, so it's approximated
    with relevance_desc sorting. Categories:
      - trending      -> sort_by=relevance_desc
      - popular       -> sort_by=popularity_desc
      - top_rated     -> special-cased below (see note)
      - upcoming      -> sort_by=release_date_desc (future-leaning)
      - now_playing   -> sort_by=release_date_desc

    NOTE on top_rated: /list-titles/ has no rating-based sort_by value and
    its results don't carry a rating field at all — only
    /title/{id}/details/ does. So "top rated" is built by pulling a larger
    popularity-sorted candidate pool, fetching details for each (which also
    gives us posters for free), and sorting locally by user_rating. This
    always costs 1 extra credit per candidate, regardless of include_posters.
    """
    sort_map = {
        "trending": "relevance_desc",
        "popular": "popularity_desc",
        "upcoming": "release_date_desc",
        "now_playing": "release_date_desc",
    }

    try:
        if category == "top_rated":
            pool_size = min(max(limit * 3, 40), 60)
            data = await watchmode_get(
                "/list-titles/",
                {"types": "movie", "sort_by": "popularity_desc", "page": 1, "limit": pool_size},
            )
            cards = await watchmode_cards_from_results(data.get("titles", []), limit=pool_size)
            cards = await enrich_cards_with_posters(cards)  # also fills user_rating
            rated = [c for c in cards if c.user_rating is not None]
            rated.sort(key=lambda c: c.user_rating, reverse=True)
            return rated[:limit]

        if category not in sort_map:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await watchmode_get(
            "/list-titles/",
            {
                "types": "movie",
                "sort_by": sort_map[category],
                "page": 1,
                "limit": limit,
            },
        )
        cards = await watchmode_cards_from_results(data.get("titles", []), limit=limit)
        if include_posters:
            cards = await enrich_cards_with_posters(cards)
        return cards

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- WATCHMODE KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/watchmode/search")
async def watchmode_search(
    query: str = Query(..., min_length=1),
    include_posters: bool = Query(
        True, description="Fetch posters via extra detail calls (costs 1 credit/title)."
    ),
):
    """
    Returns Watchmode's raw-ish search shape with 'title_results' (movies only).
    Streamlit will use it for:
      - dropdown suggestions
      - grid results

    Watchmode's /search/ endpoint often returns 'image_url' as null even for
    real movies, so we backfill missing posters via the same detail-fetch
    enrichment used for /home and /recommend/genre.
    """
    data = await watchmode_search_movies(query=query, movies_only=True)

    if include_posters:
        results = data.get("title_results", []) or []
        cards = [
            WatchmodeMovieCard(
                watchmode_id=int(r["id"]),
                title=r.get("name") or "",
                poster_url=r.get("image_url"),
                year=r.get("year"),
            )
            for r in results
            if r.get("id") is not None
        ]
        cards = await enrich_cards_with_posters(cards)
        poster_by_id = {c.watchmode_id: c.poster_url for c in cards}
        for r in results:
            if r.get("id") is not None:
                r["image_url"] = poster_by_id.get(int(r["id"])) or r.get("image_url")

    return data


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{watchmode_id}", response_model=WatchmodeMovieDetails)
async def movie_details_route(watchmode_id: int):
    return await watchmode_movie_details(watchmode_id)


# ---------- STREAMING SOURCES (Watchmode-exclusive feature) ----------
@app.get("/movie/id/{watchmode_id}/sources", response_model=List[WatchmodeSource])
async def movie_sources_route(
    watchmode_id: int,
    region: str = Query("US"),
):
    """
    Where a title can be streamed/rented/bought, by region.
    This has no TMDB equivalent in the original app — it's the main reason
    to use Watchmode.
    """
    return await watchmode_title_sources(watchmode_id, region=region)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[WatchmodeMovieCard])
async def recommend_genre(
    watchmode_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
    include_posters: bool = Query(
        True, description="Fetch posters via extra detail calls (costs 1 credit/title)."
    ),
):
    """
    Given a Watchmode title ID:
    - fetch details
    - pick first genre id
    - list movies in that genre (popularity_desc)
    """
    details = await watchmode_movie_details(watchmode_id)
    if not details.genre_ids:
        return []

    genre_id = details.genre_ids[0]
    discover = await watchmode_get(
        "/list-titles/",
        {
            "types": "movie",
            "genres": genre_id,
            "sort_by": "popularity_desc",
            "page": 1,
            "limit": limit + 1,  # pad in case we filter out the source title
        },
    )
    cards = await watchmode_cards_from_results(discover.get("titles", []), limit=limit + 1)
    cards = [c for c in cards if c.watchmode_id != watchmode_id][:limit]
    if include_posters:
        cards = await enrich_cards_with_posters(cards)
    return cards


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (Watchmode) + posters

    NOTE:
    - It selects the BEST match from Watchmode for the given query.
    - If you want MULTIPLE matches, use /watchmode/search
    """
    best = await watchmode_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No Watchmode movie found for query: {query}"
        )

    watchmode_id = int(best["id"])
    details = await watchmode_movie_details(watchmode_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_watchmode_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, watchmode=card))

    # 2) Genre recommendations (Watchmode list-titles by first genre)
    genre_recs: List[WatchmodeMovieCard] = []
    if details.genre_ids:
        genre_id = details.genre_ids[0]
        discover = await watchmode_get(
            "/list-titles/",
            {
                "types": "movie",
                "genres": genre_id,
                "sort_by": "popularity_desc",
                "page": 1,
                "limit": genre_limit + 1,
            },
        )
        cards = await watchmode_cards_from_results(
            discover.get("titles", []), limit=genre_limit + 1
        )
        genre_recs = [c for c in cards if c.watchmode_id != details.watchmode_id][:genre_limit]
        genre_recs = await enrich_cards_with_posters(genre_recs)

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


# ---------- REFERENCE LOOKUPS ----------
@app.get("/genres")
async def list_genres():
    return await watchmode_get("/genres/", {})


@app.get("/regions")
async def list_regions():
    return await watchmode_get("/regions/", {})