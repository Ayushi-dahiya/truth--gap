# TruthGap

## Corpus-Honest Retrieval-Augmented Generation

TruthGap is a Retrieval-Augmented Generation (RAG) system designed for sparse, fragmented, and potentially contradictory institutional document corpora.

Unlike a conventional RAG system that always attempts to answer a query, TruthGap evaluates the quality and sufficiency of retrieved evidence before generating an answer.

When sufficient grounded evidence is unavailable, the system abstains instead of producing an unsupported response.

## Core Components

- PDF document ingestion
- Semantic document chunking
- Dense vector retrieval
- BM25 sparse retrieval
- Hybrid retrieval with Reciprocal Rank Fusion
- Retrieval confidence scoring
- Confidence-based abstention
- LLM-based grounded generation
- Source citations
- Quantitative evaluation

## Project Goal

The primary goal is not simply to build a RAG chatbot, but to measure where and how a RAG system fails and whether it can reliably recognize when the available corpus does not support an answer.

## Team

- Sujata Pareek
- Ayushi Dahiya
- Vishwamitra Sabharwal