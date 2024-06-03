import json
import math
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from core.embeddings import EmbeddingSearcher
from core.query import QueryHandler
import ir_datasets
import logging
from core.vsm_visualizer import VectorSpaceModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# VectorSpaceModel and initialize the QueryHandler
vsm = None
query_handler = None
embed = None
suggestions_embed = None

def initialize_dataset(dataset):

    if dataset == 'antique-dataset':
        load_antique()
    elif('antique-embed'):
        print("Loading antique embed")
        load_antique_embed()
    else:
        load_wiki()


def load_wiki():
    global query_handler
    global vsm
    global suggestions_embed
    vsm = VectorSpaceModel.load("wiki_state.pkl")
    query_handler = QueryHandler(vsm)
    suggestions_embed = EmbeddingSearcher.load("wiki_embed_suggestions.pkl")

def load_antique():
    global query_handler
    global vsm
    global suggestions_embed
    vsm = VectorSpaceModel.load("antique_state.pkl")
    query_handler = QueryHandler(vsm)
    suggestions_embed = EmbeddingSearcher.load("antique_embed_suggestions.pkl")

def load_antique_embed():
    global embed
    global suggestions_embed
    embed = EmbeddingSearcher.load("antique_embed.pkl")
    suggestions_embed = EmbeddingSearcher.load("antique_embed_suggestions.pkl")


def search(query):
    if embed is not None:
        print("Searching embed")
        search_results_data = embed.search(
            query, similarity_threshold=0.25)[:100]
    else:
        search_results_data = query_handler.search(
            query, similarity_threshold=0.25)[:100]

    results = [{"result": document[1], "similarity": similarity}
               for (document, similarity) in search_results_data]
    return results

def search_suggestions(query):
    search_results_data = suggestions_embed.search(
        query, similarity_threshold=0.25)[:30]
    results = [document[1]
               for (document, similarity) in search_results_data]
    return results

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")


@app.get("/")
def mainpage(request: Request):
    return templates.TemplateResponse("select_dataset.html", {"request": request})


@app.get("/search-page")
def searchPage(request: Request):
    return templates.TemplateResponse("search_page.html", {"request": request})


@app.get("/choose-dataset")
def chooseDataset(dataset: str):
    initialize_dataset(dataset)
    response = RedirectResponse(url=f"/search-page")
    return response


@app.get("/search-results", response_class=HTMLResponse)
async def search_results(request: Request, query: str):
    return templates.TemplateResponse("search_results.html",
                                     {"request": request, "results": search(query)})


@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation(request: Request, query_id: str = None):
    with open("ir_measures.json", "r") as file:
        results = json.load(file)

    if query_id:
        results = [result for result in results if result["query_id"] == query_id]

    return templates.TemplateResponse("evaluation.html", {"request": request, "results": results, "query_id": query_id})


@app.get("/giveSuggestions", response_class=JSONResponse)
async def give_suggestions(query: str):
    # Filter suggestions based on query
    print(search_suggestions(query))
    return JSONResponse(search_suggestions(query))
