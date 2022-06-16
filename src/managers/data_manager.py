from config import db_config
import psycopg2
import logging
logging.basicConfig(level=logging.INFO)


class Manager():
    def __init__(self) -> None:
        self.__db_con = self.connect_db(db=db_config['db'],
                                        user=db_config['user'],
                                        pswd=db_config['password'],
                                        host=db_config['host'],
                                        port=db_config['port'])
        self.__cursor = self.__db_con.cursor()
        logging.info('db conllection inited')

    def connect_db(self, db: str, user: str, pswd: str, host: str, port: str):
        return psycopg2.connect(database=db,
                                user=user,
                                password=pswd,
                                host=host,
                                port=port)

    def close_connection(self):
        self.__db_con.close()
        logging.info('db conllection closed')

    def get_connection(self):
        return self.__db_con

    def get_cursor(self):
        return self.__cursor
