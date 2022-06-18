import os

db_config = {
    'db':  os.environ.get('POSTGRES_DB'),
    'user': os.environ.get('POSTGTES_USER'),
    'password': os.environ.get('POSTGRES_PASSWORD'),
    'host': os.environ.get('POSTGRES_HOST'),
    'port': os.environ.get('POSTGRES_PORT')
}

tg_config = {
    'token': os.environ.get('TG_TOKEN'),
    'endpoint': 'https://api.telegram.org/bot'
}
