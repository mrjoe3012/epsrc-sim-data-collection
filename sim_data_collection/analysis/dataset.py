import sqlite3
from typing import Tuple, Any, List
from rclpy.serialization import deserialize_message
from ugrdv_msgs.msg import VCUStatus, DriveRequest
from eufs_msgs.msg import CarState

class Dataset:
    def __init__(self):
        """
        Encapsulates an sqlite3 dataset.
        """
        self._connection = None
        self._open = False
        self._iterator = None

    def __iter__(self):
        assert self._open
        return iter(self._iterator)

    def open(self, path):
        """
        Open an sqlite3 database.
        
        :param path: Path to the sqlite3 database.
        """
        assert not self._open
        self._connection = sqlite3.connect(path)
        self._open = True
        self._iterator = DatasetIterator(self)

    def close(self):
        """
        Close the existing sqlite3 database connection.
        """
        assert self._open
        try:
            self._connection.close()
        finally:
            self._open = False
            self._iterator = None

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

    def get_msgs_short(self, msg_id):
        """
        Like get_msgs but only returns the timestamp and data.
        """
        assert self._open
        c = self._connection.cursor()
        sql = f"SELECT timestamp, data FROM {msg_id}"
        c.execute(sql)
        return c

    def get_start_and_end(self) -> Tuple[float, float]:
        """
        Get the first and last timestamps in the bag.
        :returns: A tuple containing the first and last timestamps.
        """
        states = self._connection.execute("SELECT timestamp, data FROM ground_truth_state ORDER BY timestamp ASC").fetchall()
        return (states[0][0], states[-1][0])

class DatasetIterator:
    def __init__(self, dataset: Dataset):
        """
        Encapsulates the iterator logic for an sqlite3 dataset.
        :param dataset: The dataset to create an iterator for.
        """
        self._data = self._get_data(dataset)

    def __iter__(self):
        return iter(self._data)

    def _get_data(self, dataset: Dataset) -> List[Tuple[type, float, Any]]:
        """
        Fetches and deserializes messages in the database and sorts them by timestamp.
        :param dataset: The dataset to use.
        :returns: The dataset as a sorted array.
        """
        data = []
        msg_names_and_types = [
            ("vcu_status", VCUStatus),
            ("drive_request", DriveRequest),
            ("ground_truth_state", CarState)
        ]
        for name, type in msg_names_and_types:
            data.extend([
                (type, timestamp, deserialize_message(data, type)) for timestamp, data in dataset.get_msgs_short(name).fetchall()
            ])
        data.sort(key=lambda x: x[1])
        return data
