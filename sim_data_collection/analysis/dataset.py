import sqlite3

class Dataset:
    def __init__(self):
        """
        Encapsulates an sqlite3 dataset.
        """
        self._connection = None
        self._open = False

    def open(self, path):
        """
        Open an sqlite3 database.
        
        :param path: Path to the sqlite3 database.
        """
        assert not self._open
        self._connection = sqlite3.connect(path)
        self._open = True

    def close(self):
        """
        Close the existing sqlite3 database connection.
        """
        assert self._open
        try:
            self._connection.close()
        finally:
            self._open = False

    def get_msgs(self, msg_id):
        """
        Fetch a set of messages from the database.
        
        :param msg_id: The message's string key, corresponds to the table name.
        :returns: sqlite3 Cursor object.
        """
        assert self._open
        c = self._connection.cursor()
        sql = f"SELECT * FROM {msg_id};"
        c.execute(sql)
        return c
