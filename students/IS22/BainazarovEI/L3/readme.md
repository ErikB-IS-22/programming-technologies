# Лабораторная работа №3  
**Тема:** Семантический поиск по текстам с использованием Milvus, эмбеддинг-модели `intfloat/multilingual-e5-base` и Django Rest Framework  
**Группа:** 22  
**Вариант:** 4  

---

## 1. Цель работы

1. Освоить развёртывание окружения разработки в виде devcontainer’а (Docker + VS Code Remote Containers).
2. Научиться работать с векторной БД Milvus:
   - создание коллекций,
   - загрузка эмбеддингов,
   - выполнение семантического поиска.
3. Реализовать API на Django Rest Framework для доступа к семантическому поиску.
4. Настроить запуск эмбеддинг-модели `intfloat/multilingual-e5-base` на GPU (CUDA) в контейнере.

---

## 2. Используемое программное обеспечение

- **ОС хоста:** Windows (WSL2)
- **WSL2 дистрибутив:** Ubuntu
- **Контейнеризация:** Docker Desktop (WSL2 backend)
- **Среда разработки:** Visual Studio Code + Dev Containers
- **Бэкенд:**
  - Python 3.12 (образ `mcr.microsoft.com/devcontainers/python:1-3.12-bullseye`)
  - Milvus Standalone v2.6.4
  - etcd, MinIO, Attu (web-интерфейс к Milvus)
- **ML-модель:** `intfloat/multilingual-e5-base` (`sentence-transformers`)
- **Веб-фреймворк:** Django 5.x + Django Rest Framework
- **GPU:** NVIDIA GeForce RTX 5060 Ti (CUDA 12.x, драйвер с поддержкой WSL)

---

## 3. Подготовка окружения

### 3.1. Включение WSL2 и установка Ubuntu

1. Включены компоненты Windows:
   - «Подсистема Windows для Linux»
   - «Платформа виртуальной машины»
2. Установка дистрибутива:

   ```powershell
   wsl --install -d Ubuntu
   ```

3. После перезагрузки настроен пользователь в Ubuntu.

### 3.2. Установка Docker Desktop и интеграция с WSL2

1. Установлен Docker Desktop для Windows.
2. В настройках Docker Desktop:
   - включён **Use the WSL 2 based engine**;
   - в разделе **Resources → WSL integration** включена интеграция с дистрибутивом `Ubuntu`.

Проверка в WSL:

```bash
docker --version
docker ps
```

### 3.3. VS Code и Dev Containers

1. Установлен VS Code.
2. Установлены расширения:
   - `ms-vscode-remote.remote-wsl`
   - `ms-vscode-remote.remote-containers`
3. Открыта папка проекта в режиме WSL (`WSL: Ubuntu`) и далее — **Reopen in Container** по конфигурации `.devcontainer`.

---

## 4. Структура проекта

Основные файлы/модули для лабы:

- `.devcontainer/devcontainer.json` — описание devcontainer’а.
- `.devcontainer/Dockerfile` — образ Python-окружения.
- `.devcontainer/docker-compose.yml` — запуск сервисов:
  - `app` (наш код),
  - `etcd`,
  - `minio`,
  - `standalone` (Milvus),
  - `attu`.
- `milvus_client.py` — обёртка над Milvus (создание коллекций, вставка, поиск).
- `text_parser.py` — чтение и разбиение текстов на чанки.
- `embedder.py` — работа с моделью `multilingual-e5-base` (CPU/GPU).
- `document_processor.py` — пакетная обработка текстовых файлов и загрузка в Milvus.
- `example_usage.py` — демонстрационный сценарий:
  - создание коллекции,
  - индексирование файлов,
  - семантический поиск.
- Django-часть:
  - `milvus_api/settings.py`
  - `milvus_api/urls.py`
  - `api/serializers.py`
  - `api/views.py`

---

## 5. Реализация базового сценария (CPU)

### 5.1. MilvusClient

`milvus_client.py` реализует класс `MilvusClient`, инкапсулирующий работу с Milvus:

Основные методы:

- `create_collection(collection_name, dimension, description, metric_type)`  
  Создаёт коллекцию с полями:

  - `id: INT64 (PRIMARY, auto_id)`
  - `text: VARCHAR`
  - `embedding: FLOAT_VECTOR(dim = dimension)`
  - `file_name: VARCHAR`
  - `file_path: VARCHAR`
  - `chunk_index: INT64`

- `insert_data(collection_name, texts, embeddings, file_names, file_paths, chunk_indices)`  
  Выполняет вставку батча документов/чанков.
- `search(collection_name, query_vectors, top_k)`  
  Выполняет семантический поиск по векторному полю `embedding`.

### 5.2. TextParser

`text_parser.py` реализует:

- Чтение текстового файла.
- Нормализацию пробелов.
- Разбиение текста на чанки фиксированного размера (`chunk_size`, `chunk_overlap`) с учётом переноса по пробелам/пунктуации.

Интерфейс:

```python
parser = TextParser(chunk_size=256, chunk_overlap=64)
chunks = parser.parse_file("/path/to/file.txt")
```

### 5.3. Embedder (CPU/GPU-ready)

Класс `Embedder`:

```python
from typing import List
from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Загрузка модели {model_name}...")
        print(f"Устройство: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Модель загружена. Размерность embeddings: {self.dimension}")

    def encode(self, texts: List[str], normalize: bool = True, show_progress: bool = True) -> List[List[float]]:
        prefixed_texts = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_query(self, query: str, normalize: bool = True) -> List[float]:
        prefixed_query = f"query: {query}"
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embedding.tolist()
```

Особенности:

- Автоматический выбор устройства: `cuda`, если доступна; иначе `cpu`.
- Префиксы `passage:` / `query:` как рекомендовано для `multilingual-e5-base`.

### 5.4. DocumentProcessor и example_usage.py

`DocumentProcessor`:

- получает на вход:
  - `milvus_client`,
  - `TextParser`,
  - функцию эмбеддинга (`Embedder.encode` или `create_embedding_function`),
- обходит директорию с `.txt` файлами (`/workspaces/files`),
- читает и разбивает каждый файл на чанки,
- считает эмбеддинги,
- загружает всё в коллекцию Milvus.

`example_usage.py`:

1. Подключение к Milvus.
2. Инициализация `Embedder`.
3. Создание обработчика документов.
4. Создание коллекции `documents`.
5. Обработка файлов:
   - `python_programming.txt`
   - `vector_databases.txt`
   - `machine_learning.txt`
   - `web_development.txt`
6. Семантический поиск по запросу  
   **«Что такое машинное обучение?»**.
7. Вывод топ-результатов с текстами и расстояниями.
8. Вывод информации о коллекции (количество записей).

Фактический запуск:

```bash
python example_usage.py
```

Результат (фрагмент):

- Коллекция `documents` создана.
- Заиндексировано 118 чанков.
- По запросу «Что такое машинное обучение?» найдены наиболее релевантные чанки из `machine_learning.txt`.

---

## 6. Реализация API на Django Rest Framework

### 6.1. Инициализация проекта

Внутри контейнера:

```bash
django-admin startproject milvus_api .
python manage.py startapp api
```

В `milvus_api/settings.py` добавлены приложения:

```python
INSTALLED_APPS = [
    # стандартные приложения Django...
    "rest_framework",
    "api",
]
```

### 6.2. Сериализаторы (`api/serializers.py`)

```python
from rest_framework import serializers


class CreateCollectionSerializer(serializers.Serializer):
    name = serializers.CharField()
    description = serializers.CharField(required=False, allow_blank=True)
    dimension = serializers.IntegerField()
    metric_type = serializers.ChoiceField(
        choices=["COSINE", "L2", "IP"],
        default="COSINE",
    )


class DocumentSerializer(serializers.Serializer):
    text = serializers.CharField()
    file_name = serializers.CharField(required=False, allow_blank=True)
    file_path = serializers.CharField(required=False, allow_blank=True)


class SearchSerializer(serializers.Serializer):
    query = serializers.CharField()
    collection_name = serializers.CharField()
    top_k = serializers.IntegerField(default=5)
```

### 6.3. Views (`api/views.py`)

Основные эндпоинты:

- `POST /collections/` — создать коллекцию в Milvus.
- `POST /documents/?collection=<name>` — загрузить тексты (с разбиением на чанки и эмбеддингами).
- `POST /search/` — выполнить семантический поиск.

Фрагмент `views.py`:

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import CreateCollectionSerializer, DocumentSerializer, SearchSerializer
from milvus_client import MilvusClient
from embedder import Embedder
from text_parser import TextParser

milvus = MilvusClient()
embedder = Embedder()
parser = TextParser()


class CollectionView(APIView):
    def post(self, request):
        serializer = CreateCollectionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        milvus.create_collection(
            collection_name=data["name"],
            dimension=data["dimension"],
            description=data.get("description", ""),
            metric_type=data.get("metric_type", "COSINE"),
        )
        return Response({"status": "ok"}, status=status.HTTP_201_CREATED)


class DocumentsView(APIView):
    def post(self, request):
        collection_name = request.query_params.get("collection")
        if not collection_name:
            return Response(
                {"detail": "query-параметр 'collection' обязателен"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        many = isinstance(request.data, list)
        serializer = DocumentSerializer(data=request.data, many=many)
        serializer.is_valid(raise_exception=True)
        docs = serializer.validated_data

        texts, file_names, file_paths, chunk_indices = [], [], [], []

        for doc in (docs if many else [docs]):
            chunks = parser.chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                file_names.append(doc.get("file_name", ""))
                file_paths.append(doc.get("file_path", ""))
                chunk_indices.append(i)

        embeddings = embedder.encode(texts)
        milvus.insert_data(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            file_names=file_names,
            file_paths=file_paths,
            chunk_indices=chunk_indices,
        )
        return Response({"inserted": len(texts)}, status=status.HTTP_201_CREATED)


class SearchView(APIView):
    def post(self, request):
        serializer = SearchSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        query_vec = embedder.encode_query(data["query"])
        results = milvus.search(
            collection_name=data["collection_name"],
            query_vectors=[query_vec],
            top_k=data["top_k"],
        )

        hits = results[0] if results else []
        formatted = [
            {
                "id": hit.get("id"),
                "distance": hit.get("distance"),
                "text": hit.get("text"),
                "file_name": hit.get("file_name"),
                "file_path": hit.get("file_path"),
                "chunk_index": hit.get("chunk_index"),
            }
            for hit in hits
        ]
        return Response({"results": formatted})
```

### 6.4. Маршрутизация (`milvus_api/urls.py`)

```python
from django.contrib import admin
from django.urls import path
from api.views import CollectionView, DocumentsView, SearchView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("collections/", CollectionView.as_view()),
    path("documents/", DocumentsView.as_view()),
    path("search/", SearchView.as_view()),
]
```

### 6.5. Запуск API и тестирование

Запуск сервера:

```bash
python manage.py migrate
python manage.py runserver 0.0.0.0:8001
```

Тестирование:

1. **Создание коллекции:**

   `POST http://localhost:8001/collections/`

   ```json
   {
     "name": "documents",
     "description": "Моя коллекция",
     "dimension": 768,
     "metric_type": "COSINE"
   }
   ```

2. **Добавление документов:**

   `POST http://localhost:8001/documents/?collection=documents`

   ```json
   [
     {
       "text": "Машинное обучение — это раздел искусственного интеллекта, который изучает методы обучения алгоритмов на данных.",
       "file_name": "api_test.txt",
       "file_path": "/tmp/api_test.txt"
     },
     {
       "text": "Векторные базы данных используются для хранения и поиска эмбеддингов, например Milvus.",
       "file_name": "api_test.txt",
       "file_path": "/tmp/api_test.txt"
     }
   ]
   ```

   Ответ:

   ```json
   { "inserted": <количество_чанков> }
   ```

3. **Семантический поиск:**

   `POST http://localhost:8001/search/`

   ```json
   {
     "query": "что такое машинное обучение?",
     "collection_name": "documents",
     "top_k": 3
   }
   ```

   Ответ: список наиболее похожих текстовых фрагментов с расстояниями.

---

## 7. Настройка GPU-варианта

### 7.1. Проверка GPU в WSL2

В чистом WSL:

```bash
nvidia-smi
```

Показывается видеокарта RTX 5060 Ti, драйвер и версия CUDA (12.x), что подтверждает корректную поддержку WSL-GPU.

### 7.2. Правки Dockerfile

В `.devcontainer/Dockerfile` после установки зависимостей:

```dockerfile
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip     pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Установка PyTorch с поддержкой CUDA
RUN --mount=type=cache,target=/root/.cache/pip     pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

### 7.3. Проброс GPU через docker-compose

В `.devcontainer/docker-compose.yml` для сервиса `app` добавлены:

```yaml
  app:
    build:
      context: ..
      dockerfile: .devcontainer/Dockerfile
    container_name: milvus-lab-app
    working_dir: /workspaces
    volumes:
      - ../:/workspaces
    command: sleep infinity

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: ["gpu"]

    depends_on:
      - standalone
    networks:
      - internal-network
```

### 7.4. Пересборка devcontainer и проверка

Пересборка:

```text
VS Code → Ctrl+Shift+P → Dev Containers: Rebuild and Reopen in Container
```

Проверка внутри контейнера:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Ожидаемый результат: `torch.cuda.is_available() == True`, версия CUDA соответствует установленной.

При запуске:

```bash
python example_usage.py
```

в логе `Embedder` выводит:

```text
Загрузка модели intfloat/multilingual-e5-base...
Устройство: cuda
Модель загружена. Размерность embeddings: 768
...
```

что подтверждает использование GPU.

---

## 8. Выводы

В ходе выполнения лабораторной работы:

1. Было развёрнуто контейнеризованное окружение разработки на Windows с использованием WSL2, Docker Desktop и VS Code Dev Containers.
2. Реализовано взаимодействие с векторной БД Milvus:
   - создание коллекций,
   - индексирование текстовых документов с разбиением на чанки,
   - выполнение семантического поиска с помощью эмбеддинг-модели `intfloat/multilingual-e5-base`.
3. Создано API на Django Rest Framework, предоставляющее доступ к функционалу:
   - создание коллекций (`/collections/`),
   - загрузка документов (`/documents/`),
   - семантический поиск (`/search/`).
4. Настроен запуск эмбеддинг-модели на GPU в контейнере (проброс GPU через docker-compose, установка CUDA-версии PyTorch, автоматический выбор `cuda` в классе `Embedder`).

Таким образом, цель работы достигнута: реализована полноценная система семантического поиска по коллекции текстовых документов, доступная как через скрипт, так и через REST API, с возможностью ускорения вычислений за счёт GPU.
