from pymilvus import connections, Collection, utility
from typing import List, Optional

class MilvusClient:
    def __init__(self, host="localhost", port=19530, alias="default"):
        self.host = host
        self.port = port
        self.alias = alias
        self._connect()


    def _connect(self):
        """Установка подключения к Milvus."""
        try:
            connections.connect(alias=self.alias, host=self.host, port=self.port)
        except Exception as e:
            raise Exception(f"Ошибка подключения: {e}")

    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 5,
        expr: Optional[str] = None
    ) -> List[List[dict]]:
        """
        Поиск похожих векторов.
        
        Args:
            collection_name: Имя коллекции
            query_vectors: Векторы запросов
            top_k: Количество результатов для каждого запроса
            expr: Опциональное выражение для фильтрации (например, "text like '%python%'")
        
        Returns:
            Список результатов для каждого запроса
        """
        if not utility.has_collection(collection_name):
            raise ValueError(f"Коллекция '{collection_name}' не существует")
        
        if not query_vectors:
            raise ValueError("Список векторов запросов не может быть пустым")
        
        collection = Collection(collection_name)
        
        try:
            collection.load()
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить коллекцию '{collection_name}': {e}")
        
        indexes = collection.indexes
        metric_type = "COSINE"
        if indexes:
            for index in indexes:
                if index.field_name == "embedding":
                    metric_type = index.params.get("metric_type", "COSINE")
                    break
        
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": 10}  
        }
        
        results = collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "file_name", "file_path", "chunk_index"]  
        )
        
        formatted_results = []
        for result in results:
            hits = []
            for hit in result:
                hits.append({
                    "id": hit.get('id', 'N/A'),  
                    "distance": hit.get('distance', 0),  
                    "text": hit.get('text', "Нет текста"),  
                    "file_name": hit.get('file_name', 'N/A'),  
                    "file_path": hit.get('file_path', 'N/A'),  
                    "chunk_index": hit.get('chunk_index', -1)  
                })
            formatted_results.append(hits)
        
        return formatted_results