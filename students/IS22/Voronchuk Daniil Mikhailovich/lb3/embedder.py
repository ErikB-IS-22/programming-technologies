"""
Модуль для работы с эмбеддинг-моделью multilingual-e5-base.
"""

from typing import List
from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    """Класс для генерации embeddings с использованием multilingual-e5-base."""
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = None,
        batch_size: int = 32
    ):
        """
        Инициализация эмбеддера.
        
        Args:
            model_name: Название модели (по умолчанию multilingual-e5-base)
            device: Устройство для вычислений ('cuda', 'cpu' или None для автоопределения)
            batch_size: Размер батча для обработки текстов
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Определяем устройство
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Загрузка модели {model_name}...")
        print(f"Устройство: {self.device}")
        
        # Загружаем модель
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Размерность embeddings для multilingual-e5-base = 768
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"Модель загружена. Размерность embeddings: {self.dimension}")
    
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Генерация embeddings для списка текстов.
        
        Args:
            texts: Список текстов для обработки
            normalize: Нормализовать ли векторы (рекомендуется True для cosine similarity)
            show_progress: Показывать ли прогресс обработки
        
        Returns:
            Список векторов (embeddings)
        """
        if not texts:
            return []
        
        # Для multilingual-e5-base нужно добавить префикс "query: " или "passage: "
        # Для документов используем "passage: ", для запросов - "query: "
        # Здесь обрабатываем документы, поэтому используем "passage: "
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        # Генерируем embeddings
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Конвертируем в список списков (для совместимости с Milvus)
        return embeddings.tolist()
    
    def encode_query(
        self,
        query: str,
        normalize: bool = True
    ) -> List[float]:
        """
        Генерация embedding для поискового запроса.
        
        Args:
            query: Текст запроса
            normalize: Нормализовать ли вектор
        
        Returns:
            Вектор запроса
        """
        # Для запросов используем префикс "query: "
        prefixed_query = f"query: {query}"
        
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embedding.tolist()
    
    def encode_queries(
        self,
        queries: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Генерация embeddings для списка запросов.
        
        Args:
            queries: Список текстов запросов
            normalize: Нормализовать ли векторы
            show_progress: Показывать ли прогресс
        
        Returns:
            Список векторов запросов
        """
        if not queries:
            return []
        
        # Для запросов используем префикс "query: "
        prefixed_queries = [f"query: {q}" for q in queries]
        
        embeddings = self.model.encode(
            prefixed_queries,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Получение размерности embeddings."""
        return self.dimension


# Функция-обертка для совместимости с DocumentProcessor
def create_embedding_function(
    model_name: str = "intfloat/multilingual-e5-base",
    device: str = None,
    batch_size: int = 32
) -> callable:
    """
    Создание функции для генерации embeddings (для использования с DocumentProcessor).
    
    Args:
        model_name: Название модели
        device: Устройство для вычислений
        batch_size: Размер батча
    
    Returns:
        Функция, принимающая список текстов и возвращающая список векторов
    """
    embedder = Embedder(model_name=model_name, device=device, batch_size=batch_size)
    
    def embedding_function(texts: List[str]) -> List[List[float]]:
        return embedder.encode(texts)
    
    return embedding_function


# Пример использования
if __name__ == "__main__":
    # Инициализация эмбеддера
    embedder = Embedder()
    
    # Пример текстов
    texts = [
        "Python - это язык программирования",
        "Машинное обучение использует алгоритмы",
        "Векторные базы данных хранят embeddings"
    ]
    
    # Генерация embeddings для документов
    print("\nГенерация embeddings для документов...")
    embeddings = embedder.encode(texts)
    print(f"Сгенерировано {len(embeddings)} embeddings")
    print(f"Размерность каждого embedding: {len(embeddings[0])}")
    
    # Генерация embedding для запроса
    print("\nГенерация embedding для запроса...")
    query = "Что такое Python?"
    query_embedding = embedder.encode_query(query)
    print(f"Размерность embedding запроса: {len(query_embedding)}")
    
    # Пример использования функции-обертки
    print("\nИспользование функции-обертки...")
    embedding_func = create_embedding_function()
    test_embeddings = embedding_func(["Тестовый текст"])
    print(f"Результат: {len(test_embeddings)} embedding размерностью {len(test_embeddings[0])}")