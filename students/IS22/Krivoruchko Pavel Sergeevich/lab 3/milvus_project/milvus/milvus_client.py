# milvus_project/milvus/milvus_client.py

from pymilvus import connections, Collection, utility
from pymilvus import DataType
from typing import List

class MilvusClient:
    def __init__(self, host: str = "localhost", port: int = 19530, alias: str = "default"):
        self.host = host
        self.port = port
        self.alias = alias
        self._connect()

    def _connect(self):
        """Подключение к Milvus"""
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port
            )
            print(f"Подключение к Milvus установлено ({self.host}:{self.port})")
        except Exception as e:
            print(f"Ошибка подключения к Milvus: {e}")
            raise

    def disconnect(self):
        """Отключение от Milvus"""
        connections.disconnect(self.alias)
        print("Подключение закрыто")

    def create_collection(self, collection_name: str, dimension: int) -> Collection:
        """Создание коллекции в Milvus"""
        if utility.has_collection(collection_name):
            print(f"Коллекция '{collection_name}' уже существует")
            return Collection(collection_name)

        fields = [
            {
                "name": "id",
                "dtype": DataType.INT64,
                "is_primary": True,
                "auto_id": True,
            },
            {
                "name": "embedding",
                "dtype": DataType.FLOAT_VECTOR,
                "dim": dimension,
            },
            {
                "name": "text",
                "dtype": DataType.VARCHAR,
                "max_length": 65535,
            },
            {
                "name": "file_name",
                "dtype": DataType.VARCHAR,
                "max_length": 512,
            },
            {
                "name": "chunk_index",
                "dtype": DataType.INT64,
            },
        ]

        schema = Collection.schema(fields)
        collection = Collection(name=collection_name, schema=schema, using=self.alias)

        print(f"Коллекция '{collection_name}' создана с размерностью {dimension}")
        return collection

    def insert_data(self, collection_name: str, texts: List[str], embeddings: List[List[float]]) -> List[int]:
        """Вставка данных в коллекцию"""
        if not utility.has_collection(collection_name):
            raise ValueError(f"Коллекция '{collection_name}' не существует")

        collection = Collection(collection_name)
        entities = [
            texts,
            embeddings,
        ]
        insert_result = collection.insert(entities)
        collection.flush()

        print(f"Вставлено {len(texts)} записей в коллекцию '{collection_name}'")
        return insert_result.primary_keys

    def search(self, collection_name: str, query_vectors: List[List[float]], top_k: int = 5) -> List[dict]:
        """Поиск по коллекции"""
        if not utility.has_collection(collection_name):
            raise ValueError(f"Коллекция '{collection_name}' не существует")

        collection = Collection(collection_name)
        collection.load()

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }

        results = collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["id", "text", "file_name", "chunk_index"],  
        )

        return [
            {"id": hit.id, "distance": hit.distance, 
             "text": hit.entity.get("text", "Нет текста"),
             "file_name": hit.entity.get("file_name", "N/A"),
             "chunk_index": hit.entity.get("chunk_index", -1)}
            for hit in results[0]
        ]
