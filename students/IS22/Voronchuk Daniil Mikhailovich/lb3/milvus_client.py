from pymilvus import connections, Collection, utility
from typing import List, Optional


class MilvusClient:
    """
    Клиент для работы с Milvus.
    Работает с локальным приложением (Django) + Milvus в Docker.
    """

    def __init__(self, host: str = "localhost", port: int = 19530, alias: str = "default"):
        self.host = host
        self.port = port
        self.alias = alias
        self._connect()

    # -----------------------------
    # Подключение
    # -----------------------------
    def _connect(self):
        """Установка подключения к Milvus."""
        try:
            connections.connect(alias=self.alias, host=self.host, port=self.port)
            print(f"[Milvus] Подключение установлено: {self.host}:{self.port}")
        except Exception as e:
            raise Exception(f"Ошибка подключения к Milvus: {e}")

    def disconnect(self):
        """Закрытие подключения."""
        connections.disconnect(self.alias)

    # -----------------------------
    # Поиск
    # -----------------------------
    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 5,
        expr: Optional[str] = None
    ) -> List[List[dict]]:
        """
        Поиск похожих векторов в коллекции.
        """

        if not utility.has_collection(collection_name):
            raise ValueError(f"Коллекция '{collection_name}' не существует в Milvus")

        if not query_vectors:
            raise ValueError("Список векторов запросов пуст")

        # Загружаем коллекцию
        try:
            collection = Collection(collection_name)
            collection.load()
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить коллекцию '{collection_name}': {e}")

        # Определяем метрику
        metric_type = "COSINE"
        indexes = collection.indexes

        for idx in indexes:
            if idx.field_name == "embedding":
                metric_type = idx.params.get("metric_type", "COSINE")

        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": 10}
        }

        # Выполняем поиск
        try:
            results = collection.search(
                data=query_vectors,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["text", "file_name", "file_path", "chunk_index"]
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка поиска в Milvus: {e}")

        # Форматируем вывод
        formatted_results = []

        for result in results:
            hits_list = []

            for hit in result:
                hits_list.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "text": hit.entity.get("text", ""),
                    "file_name": hit.entity.get("file_name", ""),
                    "file_path": hit.entity.get("file_path", ""),
                    "chunk_index": hit.entity.get("chunk_index", -1)
                })

            formatted_results.append(hits_list)

        return formatted_results
