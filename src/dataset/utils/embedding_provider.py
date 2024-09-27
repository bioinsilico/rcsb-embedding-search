import os.path
import sqlite3
import json
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AbstractEmbeddingProvider(ABC):

    @abstractmethod
    def build(self, root_folder, local_rank):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def set(self, key, numbers_list):
        pass

    @abstractmethod
    def update(self, key, new_numbers):
        pass


class SqliteEmbeddingProvider(AbstractEmbeddingProvider):

    def __init__(self):
        """Initialize the database connection and create the table."""
        self.collection_name = None
        self.cursor = None
        self.conn = None
        self.db_name = None

    def build(self, root_folder, global_rank):
        directory = os.path.join(root_folder, global_rank)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.db_name = self.db_name = os.path.join(directory, "embeddings.sqlite")
        self.collection_name = f"embedding_{global_rank}"
        self.connect()
        self.create_table()

    def connect(self):
        """Connect to the SQLite database."""
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def create_table(self):
        """Remove the collection table if it exists, then create a new one."""
        # Drop the table if it exists
        self.cursor.execute(f'DROP TABLE IF EXISTS {self.collection_name}')
        # Create a new table
        self.cursor.execute(f'''
            CREATE TABLE {self.collection_name} (
                id TEXT PRIMARY KEY,
                numbers TEXT
            )
        ''')
        self.conn.commit()

    def get(self, key):
        """Retrieve the list of numbers for a given key."""
        self.cursor.execute(f'SELECT numbers FROM {self.collection_name} WHERE id = ?', (key,))
        result = self.cursor.fetchone()
        if result:
            numbers_json = result[0]
            numbers = json.loads(numbers_json)
            return numbers
        else:
            return None  # Or you can raise KeyError(key)

    def set(self, key, numbers_list):
        """Set the list of numbers for a given key."""
        numbers_json = json.dumps(numbers_list)
        self.cursor.execute(f'''
            INSERT OR REPLACE INTO {self.collection_name} (id, numbers) VALUES (?, ?)
        ''', (key, numbers_json))
        self.conn.commit()

    def update(self, key, new_numbers):
        """Replace the existing list of numbers with a new list."""
        if self.get(key) is not None:
            self.set(key, new_numbers)
        else:
            raise KeyError(f"Key '{key}' does not exist.")

    def delete(self, key):
        """Delete the entry for a given key."""
        self.cursor.execute(f'DELETE FROM {self.collection_name} WHERE id = ?', (key,))
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()


class DiskEmbeddingProvider(AbstractEmbeddingProvider):

    def __init__(self):
        self.embedding_path = None

    def build(self, root_folder, local_rank):
        directory = os.path.join(root_folder, local_rank)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.embedding_path = directory

    def get(self, key):
        return list(pd.read_csv(
            os.path.join(self.embedding_path, f"{key}.csv"),
            header=None,
            index_col=None
        ).iloc[:, 0].values)

    def set(self, key, numbers_list):
        pd.DataFrame(np.array(numbers_list)).to_csv(
            os.path.join(self.embedding_path, f"{key}.csv"),
            header=False,
            index=False
        )

    def update(self, key, new_numbers):
        self.set(key, new_numbers)
