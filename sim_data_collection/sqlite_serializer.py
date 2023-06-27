import sqlite3 as db

class SQLiteSerializer:
    def __init__(self):
        self._open = False

    def open(self, path):
        assert self._open == False
        self._connection = db.connect(path)
        self._open = True

    def close(self):
        assert self._open == True
        self._connection.close()
        self._open = False

    def create_new_database(self):
        assert self._open == True
        cursor = self._connection.cursor()
        queries = ("""
        CREATE TABLE vcu_status(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            data BLOB
        );
        """,
        """
        CREATE TABLE perception_cones(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            data BLOB
        );
        """,
        """
        CREATE TABLE ground_truth_cones(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            data BLOB
        );
        """,
        """
        CREATE TABLE ground_truth_state(
            hash VARCHAR(32) PRIMARY KEY,
            timestamp INT NOT NULL,
            data BLOB
        );
        """,
        )
        for sql in queries: cursor.execute(sql)

    def serialize_message(self, msg):
        pass

    def drop_unmet_dependencies(self):
        pass