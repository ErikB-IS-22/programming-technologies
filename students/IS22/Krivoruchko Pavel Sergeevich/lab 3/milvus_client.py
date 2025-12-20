"""
Клиент для работы с Milvus векторной базой данных.
Примеры использования для лабораторной работы по семантическому поиску.
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import List, Optional


class MilvusClient:
    """Клиент для работы с Milvus."""
    
    def __init__(
        self,
        host: str = "standalone",
        port: int = 19530,
        alias: str = "default"
    ):
        """
        Инициализация клиента Milvus.
        
        Args:
            host: Хост Milvus (в docker-compose это 'standalone')
            port: Порт Milvus (по умолчанию 19530)
            alias: Алиас подключения
        """
        self.host = host
        self.port = port
        self.alias = alias
        self._connect()
    
    def _connect(self):
        """Установка подключения к Milvus."""
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
        """Закрытие подключения."""
        connections.disconnect(self.alias)
        print("Подключение закрыто")
    
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        description: str = "",
        metric_type: str = "COSINE"
    ) -> Collection:
        """
        Создание коллекции в Milvus.
        
        Args:
            collection_name: Имя коллекции
            dimension: Размерность векторов
            description: Описание коллекции
            metric_type: Тип метрики расстояния (COSINE, L2, IP)
        
        Returns:
            Созданная коллекция
        """
        # Проверяем, существует ли коллекция
        if utility.has_collection(collection_name):
            print(f"Коллекция '{collection_name}' уже существует")
            return Collection(collection_name)
        
        # Определяем схему полей
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Автоинкрементный ID"
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Исходный текст"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dimension,
                description="Векторное представление текста"
            ),
            FieldSchema(
                name="file_name",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Название файла-источника"
            ),
            FieldSchema(
                name="file_path",
                dtype=DataType.VARCHAR,
                max_length=1024,
                description="Полный путь к файлу-источнику"
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64,
                description="Индекс чанка в документе (начинается с 0)"
            )
        ]
        
        # Создаем схему
        schema = CollectionSchema(
            fields=fields,
            description=description
        )
        
        # Создаем коллекцию
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.alias
        )
        
        # Создаем индекс для поля embedding
        index_params = {
            "metric_type": metric_type,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        print(f"Коллекция '{collection_name}' создана (dim={dimension}, metric={metric_type})")
        return collection
    
    def insert_data(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]],
        file_names: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
        chunk_indices: Optional[List[int]] = None
    ) -> List[int]:
        """
        Вставка данных в коллекцию.
        
        Args:
            collection_name: Имя коллекции
            texts: Список текстов
            embeddings: Список векторов (embeddings)
            file_names: Список названий файлов (опционально)
            file_paths: Список путей к файлам (опционально)
            chunk_indices: Список индексов чанков (опционально)
        
        Returns:
            Список ID вставленных записей
        """
        if not utility.has_collection(collection_name):
            raise ValueError(f"Коллекция '{collection_name}' не существует")
        
        if len(texts) != len(embeddings):
            raise ValueError("Количество текстов должно совпадать с количеством векторов")
        
        if not texts:
            raise ValueError("Список текстов не может быть пустым")
        
        collection = Collection(collection_name)
        
        # Проверяем размерность embeddings
        if embeddings:
            try:
                # Ищем поле embedding в схеме
                embedding_field = None
                for field in collection.schema.fields:
                    if field.name == "embedding":
                        embedding_field = field
                        break
                
                if embedding_field:
                    expected_dim = embedding_field.params['dim']
                    actual_dim = len(embeddings[0])
                    if actual_dim != expected_dim:
                        raise ValueError(
                            f"Неверная размерность embeddings: ожидается {expected_dim}, получено {actual_dim}"
                        )
            except (IndexError, KeyError) as e:
                raise ValueError(f"Ошибка при проверке размерности: {e}")
        
        try:
            collection.load() 
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить коллекцию '{collection_name}': {e}")
        
        # Подготавливаем метаданные (используем значения по умолчанию если не указаны)
        num_items = len(texts)
        if file_names is None:
            file_names = [""] * num_items
        if file_paths is None:
            file_paths = [""] * num_items
        if chunk_indices is None:
            chunk_indices = list(range(num_items))
        
        # Проверяем соответствие длин
        if len(file_names) != num_items or len(file_paths) != num_items or len(chunk_indices) != num_items:
            raise ValueError("Длины всех списков метаданных должны совпадать с количеством текстов")
        
        # Подготавливаем данные для вставки
        entities = [
            texts,
            embeddings,
            file_names,
            file_paths,
            chunk_indices
        ]
        
        # Вставляем данные
        insert_result = collection.insert(entities)
        collection.flush()  # Принудительно сохраняем изменения
        
        print(f"Вставлено {len(texts)} записей в коллекцию '{collection_name}'")
        return insert_result.primary_keys
    
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
        
        # Получаем метрику из индекса коллекции
        indexes = collection.indexes
        metric_type = "COSINE"  # По умолчанию
        if indexes:
            # Берем метрику из первого индекса поля embedding
            for index in indexes:
                if index.field_name == "embedding":
                    metric_type = index.params.get("metric_type", "COSINE")
                    break
        
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": 10}  # Количество кластеров для поиска
        }
        
        results = collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "file_name", "file_path", "chunk_index"]  # Возвращаем текст и метаданные
        )
        
        # Форматируем результаты
        formatted_results = []
        for result in results:
            hits = []
            for hit in result:
                hits.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "text": hit.entity.get("text", ""),
                    "file_name": hit.entity.get("file_name", ""),
                    "file_path": hit.entity.get("file_path", ""),
                    "chunk_index": hit.entity.get("chunk_index", -1)
                })
            formatted_results.append(hits)
        
        return formatted_results
    
    def get_collection_info(self, collection_name: str) -> dict:
        """Получение информации о коллекции."""
        if not utility.has_collection(collection_name):
            return {"exists": False}
        
        try:
            collection = Collection(collection_name)
            collection.load()
            
            return {
                "exists": True,
                "num_entities": collection.num_entities,
                "schema": collection.schema,
                "indexes": collection.indexes
            }
        except Exception as e:
            return {
                "exists": True,
                "error": f"Ошибка при получении информации: {e}"
            }
    
    def delete_collection(self, collection_name: str):
        """Удаление коллекции."""
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Коллекция '{collection_name}' удалена")
        else:
            print(f"Коллекция '{collection_name}' не существует")
    
    def get_document_chunks(
        self,
        collection_name: str,
        file_path: str,
        order_by_index: bool = True
    ) -> List[dict]:
        """
        Получение всех чанков из указанного документа.
        
        Args:
            collection_name: Имя коллекции
            file_path: Путь к файлу
            order_by_index: Сортировать ли чанки по индексу
        
        Returns:
            Список чанков документа с их данными
        """
        if not utility.has_collection(collection_name):
            raise ValueError(f"Коллекция '{collection_name}' не существует")
        
        collection = Collection(collection_name)
        try:
            collection.load()
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить коллекцию '{collection_name}': {e}")
        
        # Ищем все записи с указанным file_path
        # Экранируем кавычки в пути
        escaped_path = file_path.replace('"', '\\"')
        expr = f'file_path == "{escaped_path}"'
        
        try:
            results = collection.query(
                expr=expr,
                output_fields=["id", "text", "file_name", "file_path", "chunk_index"]
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка при запросе данных: {e}")
        
        if order_by_index:
            results.sort(key=lambda x: x.get("chunk_index", 0))
        
        return results
    
    def reconstruct_document(
        self,
        collection_name: str,
        file_path: str
    ) -> str:
        """
        Восстановление исходного документа из всех его чанков.
        
        Args:
            collection_name: Имя коллекции
            file_path: Путь к файлу
        
        Returns:
            Восстановленный текст документа
        """
        chunks = self.get_document_chunks(collection_name, file_path, order_by_index=True)
        
        if not chunks:
            return ""
        
        texts = [chunk["text"] for chunk in chunks]
        return " ".join(texts)

