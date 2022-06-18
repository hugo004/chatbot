from .data_manager import Manager
from log import logging


class FeedbackManager(Manager):
    def __init__(self):
        Manager.__init__(self)
        self.__cursor = self.get_cursor()
        self.__connection = self.get_connection()

    def save_feedback(self, sentence: str, feedback_type=-1, predicted=None):
        self.__cursor.execute(
            f'''INSERT INTO feedback(sentence, predicted, feedback) values ('{sentence}', '{predicted}', '{feedback_type}')''')
        self.__connection.commit()

        logging.info('user feedback saved')
