# Structural Cohesion in Word Embeddings

This repository contains the code and analysis for my undergraduate honors thesis in statistics: **"Distance and cohesion in word embeddings"**. The work explores a novel metric, *cohesion*, for evaluating structural relationships in word embeddings—complementing or enhancing traditional similarity metrics like Euclidean distance and cosine similarity.

## 📘 Project Summary

Traditional measures such as Euclidean and cosine similarity quantify closeness between word embeddings, but often lack insight into the *structure* or *internal consistency* of word neighborhoods. This project proposes the concept of **cohesion**, inspired by clustering theory, to characterize how tightly word vectors are grouped in the embedding space—revealing patterns missed by pairwise similarities alone.

Key contributions include:
- A mathematical definition of cohesion for word neighborhoods.
- Analysis of how cohesion relates to semantic meaning.
- Empirical results across different embedding models and datasets.
- Comparison of cohesion with cosine similarity and Euclidean distance.

## 🧠 Concepts

- **Word Embeddings**: Vector representations of words from models like GloVe, Word2Vec, and fastText.
- **Cohesion**: A novel structural metric based on internal similarity within a word’s neighborhood.
- **PALD**: Partitioned Aggregate Local Depth—used to compute cohesion.
- **Cosine Similarity vs. Euclidean Distance**: Traditional metrics contrasted with cohesion.

## 🔍 Requirements

- Python 3.8+
- contourpy==1.3.1
- cycler==0.12.1
- fonttools==4.55.6
- kiwisolver==1.4.8
- matplotlib==3.10.0
- numpy==2.2.2
- packaging==24.2
- pandas==2.2.3
- pillow==11.1.0
- pyparsing==3.2.1
- python-dateutil==2.9.0.post0
- pytz==2024.2
- six==1.17.0
- tqdm==4.67.1
- tzdata==2025.1
