# Chatbot

## Description
A chatbot with NLP

### Rerequriments
- docker
- docker-compose
- python version > 3.8

#### Start Postgres DB
```
docker compose --env-file ./config/.env up db
```
#### Start App
```
docker compose --env-file ./config/.env up app
```