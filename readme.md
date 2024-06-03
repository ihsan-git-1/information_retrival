# Information Retrieval System


An Information Retrieval System designed to efficiently store, retrieve, and manage information from large datasets.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Features
- **User Query Suggestions**
- **Model Trainning**
- **Word Embeddings**
- **Pre Processing**
- **Indexing**
- **Query Processing**
- **Matching & Ranking**
- **UI/UX**
- **SOA**

## Usage
- **Run main_vsm_build.ipynb to add create the tfidf vectorizer for each dataset**
- **Run suggestions_build_tables.ipynb to store the offline suggestions**
- **Run embedding_build.ipynb to train the model and to store the offline models**
- **Run main_antique_evaluation.ipynb to get antique evaluations with embeddings and without**
- **Run main_wiki_evaluation.ipynb to get wikiR evaluations**

## Installation

```bash
$ python3 -m venv env/python
$ source env/python/bin/activate
$ python3 -m pip install -r requirements.txt
```
