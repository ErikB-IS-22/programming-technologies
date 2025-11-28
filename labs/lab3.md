# Лабораторная работа №3. Векторные базы данных и семантический поиск

<ins>Цель</ins>: научиться использовать векторные базы данных и семантический поиск для обработки текстовых данных и потенциальной интеграции с LLM.

## План

1. Настройка окружения;
2. Создание модуля для работы с милвусом;
3. Парсинг текстовых файлов;
4. Задания.

## 1. Настройка окружения

В отличие от предыдущих лабораторных работ, данная работа будет выполняться в devcontainer. Конфиг вы можете найти по пути [assets/lab3/.devcontainer/devcontainer.json](../assets/lab3/.devcontainer/devcontainer.json).
Чем по сути своей является devcontainer? Это контейнер, который содержит всю необходимую для работы среду. В данном случае, это:
- Python 3.12;
- Milvus;
- Attu;

Devcontainer позволяет нам работать с проектом в изолированном окружении, что позволяет избежать конфликтов между зависимостями и упрощает процесс установки и настройки окружения.
Для запуска devcontainer необходимо вызвать контекстное меню в VSCode (Ctrl+Shift+P) и выбрать "Reopen in Container". Убедитесь, что у вас установлено расширение "Dev Containers".
Сразу предупрежу, что контейнеры достаточно тяжелые, поэтому не удивляйтесь, что у вас идет загрузка 2+ гигабайт данных)
Конфиг, написанный мной (ямл файл, жсон и докерфайл) подходят для использования cpu версии эмбеддера. Если вы хотите использовать gpu версию, то вам необходимо изменить конфиг.

## 2. Создание векторной базы данных

Итак, у нас работает контейнер, запущен Милвус, Атту (интерфейс для Милвуса) пашет на порте 8000, как с этим работать?

Начнем с того, что инициализируем модуль милвуса.

```python
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
```

Лично я порт захардкодил прям в ините, но вы можете сделать его подтягивающимся из .env файла.
Дальше мы создаем метод для подключения к милвусу и метод для отключения от него.

```python
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
```

Круто, теперь мы можем производить коннекшн. Но помимо подключения нужно и данные туда вгружать, это же база данных как-никак.
Напишем пару методов для работы с коллекциями.

```python
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
        if utility.has_collection(collection_name):
            print(f"Коллекция '{collection_name}' уже существует")
            return Collection(collection_name)
        
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
        
        schema = CollectionSchema(
            fields=fields,
            description=description
        )
        
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.alias
        )
        
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
```

Ух, огромный метод получился. Итак, что тут к чему вообще? `create_collection` - метод для создания коллекции в милвусе.
`collection_name` - имя коллекции, которое мы хотим создать.
`dimension` - размерность векторов, которые мы будем использовать.
`description` - описание коллекции.
`metric_type` - тип метрики расстояния, который мы будем использовать.

Метод создает коллекцию с указанными параметрами и индексом для поля `embedding`.
Внутри метода вы можете заметить интересное такое поле `chunk_index`. Это поле будет использоваться для хранения индекса чанка в документе.
Но что такое чанк? Это кусок текста, который мы будем использовать для поиска. Например, если у нас есть документ в 1000 символов, и мы хотим его разделить на чанки по 100 символов, то у нас будет 10 чанков.
Подобный подод позволяет нам более точно подходить к поиску, так как чем больше чанков, тем более точно мы сможем исходя из контекста вопроса найти нужный нам фрагмент текста, а по нему уже определить сам документ.

Теперь рассмотрим вставку данных в коллекцию.
```python
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
        
        if embeddings:
            try:
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
        
        num_items = len(texts)
        if file_names is None:
            file_names = [""] * num_items
        if file_paths is None:
            file_paths = [""] * num_items
        if chunk_indices is None:
            chunk_indices = list(range(num_items))
        
        if len(file_names) != num_items or len(file_paths) != num_items or len(chunk_indices) != num_items:
            raise ValueError("Длины всех списков метаданных должны совпадать с количеством текстов")
        
        entities = [
            texts,
            embeddings,
            file_names,
            file_paths,
            chunk_indices
        ]
        
        insert_result = collection.insert(entities)
        collection.flush()
        
        print(f"Вставлено {len(texts)} записей в коллекцию '{collection_name}'")
        return insert_result.primary_keys
```

Не меньший блок кода, не так ли? Впрочем, если рассмотреть его чуть подробнее, то в целом он перестанет быть таким страшным.
Вкратце, что тут происходит? Мы:
- Проверяем, существует ли коллекция;
- Проверяем, соответствует ли размерность embeddings;
- Загружаем коллекцию;
- Подготавливаем данные для вставки;
- Вставляем данные в коллекцию;
- Принудительно сохраняем изменения.

После этого мы можем проверить, что данные были вставлены правильно.
```python
print(f"Вставлено {len(texts)} записей в коллекцию '{collection_name}'")
print(insert_result.primary_keys)
```

Замечательно! Как по этому делу искать что-то в милвусе?

```python
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
                    "id": hit.id,
                    "distance": hit.distance,
                    "text": hit.entity.get("text", ""),
                    "file_name": hit.entity.get("file_name", ""),
                    "file_path": hit.entity.get("file_path", ""),
                    "chunk_index": hit.entity.get("chunk_index", -1)
                })
            formatted_results.append(hits)
        
        return formatted_results
```

Вкратце, что тут происходит? Мы:
- Проверяем, существует ли коллекция;
- Проверяем, соответствует ли размерность embeddings;
- Загружаем коллекцию;
- Получаем метрику из индекса коллекции;
- Выполняем поиск;
- Форматируем результаты.

По итогу на выходе у нас имеется список результатов, каждый из которых представляет собой список словарей, каждый из которых содержит информацию о найденном векторе.
Сам поиск происходит по метрике cosine similarity, но вы можете попробовать использовать и другие метрики, например, L2 или IP.

## 3. Парсинг текстовых файлов

Ну, модуль для работы с Милвусом написан, теперь нужно в него реальные данные вгружать.
С процедурой парсинга вы знакомы ещё с третьего курса, но тогда вы скорее всего парсили сайты через BeautifulSoup, а не текстовые файлы.

Рассмотрим классический такой пример парсера текстовых файлов, но с некоторыми дополнительными фичами чисто под наши нужды.

```python
from typing import List
import os
import re


class TextParser:
    """Парсер текстовых файлов с разбиением на чанки."""
    
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 64
    ):
        """
        Инициализация парсера.
        
        Args:
            chunk_size: Размер чанка в символах
            chunk_overlap: Размер перекрытия между чанками в символах
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap
```

Чанки, оверлапы, степ сайзы, что это такое?
- Чанок: кусок текста, который мы будем использовать для поиска.
- Оверлап: перекрытие между чанками. Нужно для того, чтобы чанки имели друг с другом связность.
- Степ сайз: шаг, на который мы будем перемещаться при разбиении текста на чанки.

Файл нам нужно прочитать и нормализовать.
```python
def read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """
        Чтение текстового файла.
        
        Args:
            file_path: Путь к файлу
            encoding: Кодировка файла
        
        Returns:
            Содержимое файла как строка
        
        Raises:
            FileNotFoundError: Если файл не существует
            ValueError: Если файл пустой
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        
        if not content.strip():
            raise ValueError(f"Файл '{file_path}' пуст")
        
        return content
    
    def normalize_text(self, text: str) -> str:
        """
        Нормализация текста: удаление лишних пробелов, переносов строк.
        
        Args:
            text: Исходный текст
        
        Returns:
            Нормализованный текст
        """
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
```

Нормализация текста: удаление лишних пробелов, переносов строк.

Дальше мы создаем метод для разбиения текста на чанки.

```python
def chunk_text(self, text: str) -> List[str]:
        """
        Разбиение текста на чанки с перекрытием.
        
        Args:
            text: Исходный текст
        
        Returns:
            Список чанков
        """
        text = self.normalize_text(text)
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                search_end = min(end + 50, len(text))
                chunk = text[start:search_end]
                
                last_space = chunk.rfind(' ', 0, self.chunk_size)
                last_punct = max(
                    chunk.rfind('.', 0, self.chunk_size),
                    chunk.rfind('!', 0, self.chunk_size),
                    chunk.rfind('?', 0, self.chunk_size),
                    chunk.rfind(',', 0, self.chunk_size)
                )
                
                break_point = max(last_punct, last_space)
                
                if break_point > self.chunk_size * 0.7:
                    end = start + break_point + 1
                else:
                    end = start + self.chunk_size
            else:
                end = len(text)
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = start + self.step_size
            
            if start < len(text) and start + self.step_size >= len(text):
                remaining = text[start:].strip()
                if remaining and remaining not in chunks:
                    chunks.append(remaining)
                break
        
        return chunks
```

Как видите, чанкирование представляет из себя простое разбиение текста на куски, с учетом оверлапа и степ сайза.
В целом, это достаточно простой и понятный алгоритм, который позволяет нам разбивать текст на чанки, с учетом связности и контекста.
Дальше создаем метод для полного парсинга файла.

```python
def parse_file(self, file_path: str, encoding: str = "utf-8") -> List[str]:
        """
        Парсинг файла с автоматическим чанкированием.
        
        Args:
            file_path: Путь к файлу
            encoding: Кодировка файла
        
        Returns:
            Список чанков текста
        """
        text = self.read_file(file_path, encoding)
        chunks = self.chunk_text(text)
        return chunks
```

Итогом работы данного класса является список чанков текста.
Теперь надо написать модуль для полноценной обработки N-ного количества таких .txt файлов.

```python
from typing import List, Callable, Optional
from milvus_client import MilvusClient
from text_parser import TextParser


class DocumentProcessor:
    """Обработчик документов для загрузки в векторную БД."""
    
    def __init__(
        self,
        milvus_client: MilvusClient,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None
    ):
        """
        Инициализация обработчика документов.
        
        Args:
            milvus_client: Клиент Milvus
            chunk_size: Размер чанка в символах
            chunk_overlap: Размер перекрытия между чанками
            embedding_function: Функция для генерации embeddings
                                Принимает список текстов, возвращает список векторов
        """
        self.milvus_client = milvus_client
        self.parser = TextParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_function = embedding_function
```

Что тут происходит? Мы инициализируем класс DocumentProcessor, который будет использоваться для обработки файлов. Внутри него мы инициализируем классы TextParser и MilvusClient, а также функцию для генерации embeddings. Функция для генерации embeddings будет использоваться для генерации embeddings для каждого чанка. Что такое embeddings? Это векторы, которые представляют собой числовые представления текста (не зря же мы с вами чанкер писали:D).

Далее задачей данного класса будет являться обработка всех файлов в некоторой директории, их чанкирование, генерация embeddings и загрузка в милвус. Подробно с данными методами вы можете ознакомиться в файле [document_processor.py](../assets/lab3/document_processor.py).

Теперь рассмотрим эмбеддер как таковой. В данной лабораторной работе мы будем использовать эмбеддер multilingual-e5-base, который является многоязычным эмбеддером, который может обрабатывать текст на разных языках, включая русский.

```python
class Embedder:
    """Класс для генерации embeddings с использованием multilingual-e5-base."""
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = None,
        batch_size: int = 32
    ):
```

Что тут происходит? При инициализации класса Embedder мы загружаем модель multilingual-e5-base через библиотеку sentence-transformers. Модель автоматически определяет, на каком устройстве работать (GPU или CPU), если вы не указали device явно. Размер батча по умолчанию равен 32 — это означает, что за раз мы будем обрабатывать 32 текста одновременно, что ускоряет процесс генерации embeddings.

Важный момент! multilingual-e5-base требует специальных префиксов для текстов! Для документов нужно использовать префикс "passage: ", а для поисковых запросов — "query: ". Это не просто прихоть разработчиков модели, а важная особенность архитектуры, которая позволяет модели лучше различать контекст использования текста.

```python
def encode(
    self,
    texts: List[str],
    normalize: bool = True,
    show_progress: bool = True
) -> List[List[float]]:
    prefixed_texts = [f"passage: {text}" for text in texts]
    
    embeddings = self.model.encode(
        prefixed_texts,
        batch_size=self.batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    return embeddings.tolist()
```

Метод `encode` используется для генерации embeddings документов. Он принимает список текстов, добавляет к каждому префикс "passage: ", а затем генерирует векторы. Параметр `normalize=True` нормализует векторы, что необходимо для корректной работы с косинусным расстоянием в Milvus (которое мы используем по умолчанию).

Для поисковых запросов используется отдельный метод `encode_query`:

```python
def encode_query(
    self,
    query: str,
    normalize: bool = True
) -> List[float]:
    prefixed_query = f"query: {query}"
    
    embedding = self.model.encode(
        prefixed_query,
        normalize_embeddings=normalize,
        convert_to_numpy=True
    )
    return embedding.tolist()
```

Почему отдельный метод? Потому что для запросов нужен префикс "query: ", а не "passage: ". Это позволяет модели генерировать embeddings, которые лучше подходят для семантического поиска — векторы запросов и документов будут в одном векторном пространстве, но оптимизированы для своих задач.

Размерность embeddings для multilingual-e5-base составляет 768 — это фиксированное значение, которое нужно знать при создании коллекции в Milvus (помните параметр `dimension`?).

Также в модуле есть функция-обертка `create_embedding_function`, которая создает функцию для использования с DocumentProcessor. Она просто оборачивает метод `encode` в удобный формат, который ожидает DocumentProcessor. Подробно с реализацией вы можете ознакомиться в файле [embedder.py](../assets/lab3/embedder.py).

--------------------------------

Запустить всё это дело можно командой `python example_usage.py`. В данном файлике накидан стандартный сценарий, который:
- Подключается к милвусу;
- Создает функцию для генерации embeddings;
- Создает обработчик документов;
- Обрабатывает все файлы в директории `files`;
- Создает коллекцию в милвусе;
- Выполняет семантический поиск;
- Показывает результаты.

## 4. Задания

1. Запустить описанный выше сценарий с помощью gpu. Нужно модифицировать конфиг под использование cuda-зависимостей.
2. Реализовать полноценное API для работы с милвусом. Используйте [Django Rest Framework](https://docs.djangoproject.com/en/5.2/).