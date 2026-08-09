# Recommender Systems and Reinforcement Learning

<p align="center">
  <img src="public/logo_UMONS.png" alt="University of Mons" height="100">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="public/FS.png" alt="Faculty of Science" height="100">
</p>

This repository contains the code developed as part of a Master's thesis in Computer Science at the University of Mons, Belgium. The research was conducted within the Theoretical Computer Science Laboratory and focuses on content recommendation and reinforcement learning.

The thesis first presents classical recommendation approaches based on matrix factorization, before studying reinforcement learning methods for recommendation. The work covers the mathematical and algorithmic concepts underlying these approaches and their application to content recommendation.

The repository is organized into chapters following the structure of the thesis. Chapter 5 focuses on the practical implementation of the proposed algorithms. The implementation is based on a backend API developed with FastAPI and a frontend interface built with React. The project also includes Docker configurations, Docker Compose files, and Helm charts for deployment on Kubernetes.

A live deployment of the application is available during the thesis period, allowing the system to be explored without requiring users to train the models themselves.

The work brings together concepts from statistics, game theory, optimization, machine learning, and algorithms, with a focus on their application to recommender systems and reinforcement learning.

## Getting Started

### Chapter 1

Create the `.env` file at the repository root:

```bash
cp .env.example .env
```

Add your Kaggle credentials:

| Variable | Value |
| --- | --- |
| `KAGGLE_USERNAME` | Your Kaggle username |
| `KAGGLE_API_TOKEN` | Your Kaggle API token |

The Chapter 1 directory contains additional documentation explaining how to obtain the Kaggle API credentials.

---

### Chapter 5

#### Prerequisites

- Docker
- Docker Compose

From the repository root:

```bash
cd chap_5
cp .env.example .env
```

Configure the `.env` file:

| Variable | Value |
| --- | --- |
| `POSTGRES_USER` | `postgres` |
| `POSTGRES_PASSWORD` | `postgres` |
| `POSTGRES_DB` | `master` |
| `POSTGRES_HOST` | `db` |
| `POSTGRES_PORT` | `5432` |
| `KAGGLE_USERNAME` | Your Kaggle username |
| `KAGGLE_API_TOKEN` | Your Kaggle API token |
| `MDP_K` | `3` |
| `MDP_FRACTION` | `1.0` |
| `MDP_MAX_SKIP` | `10` |
| `MDP_BATCH_SIZE` | `10000` |
| `MDP_GAMMA_BOOST` | `1.0` |
| `MDP_GAMMA_RL` | `0.9` |
| `MDP_LIST_SIZE` | `3` |
| `MDP_THRESHOLD` | `0.001` |
| `MDP_TEMPERATURE` | `1.0` |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` |

Then start the application:

```bash
docker compose up -d --build
```