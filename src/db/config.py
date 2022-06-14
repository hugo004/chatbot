import psycopg2
import logging
logging.basicConfig(level=logging.INFO)


class DB():
    def __init__(self, db_name='your db', user='user name', pswd='password', host='db host', port='port') -> None:
        self.__db_con = self.connect_db(db=db_name,
                                        user=user,
                                        pswd=pswd,
                                        host=host,
                                        port=port)
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

    def insert_feedback(self, sentence: str, feedback_type=-1, predicted=None):
        self.__cursor.execute(
            f'''INSERT INTO CAHTBOT_FEEDBACK(sentence, predicted, feedback) values ('{sentence}', '{predicted}', '{feedback_type}')''')
        self.__db_con.commit()

        logging.info('user feedback saved')
