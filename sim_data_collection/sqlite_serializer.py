import sqlite3

class SQLiteSerializer:
    def __init__(self):
        self._open = False

    def open(self, path):
        assert self._open == False
        self._open = True

    def close(self):
        assert self._open == True
        self._open = False

    def create_new_database(self):
        assert self._open == True

    def serialize_message(self, msg):
        pass

    def drop_unmet_dependencies(self):
        pass