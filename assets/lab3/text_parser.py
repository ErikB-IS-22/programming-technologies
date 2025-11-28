"""
Парсер текстовых файлов с чанкированием для семантического поиска.
"""

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
        # Заменяем множественные пробелы и переносы строк на одинарные
        text = re.sub(r'\s+', ' ', text)
        # Убираем пробелы в начале и конце
        text = text.strip()
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Разбиение текста на чанки с перекрытием.
        
        Args:
            text: Исходный текст
        
        Returns:
            Список чанков
        """
        # Нормализуем текст
        text = self.normalize_text(text)
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Определяем конец текущего чанка
            end = start + self.chunk_size
            
            # Если это не последний чанк, пытаемся разбить по границе слова
            if end < len(text):
                # Ищем ближайший пробел или знак препинания перед концом, чтобы не разрывать слова
                search_end = min(end + 50, len(text))  # Ищем в пределах 50 символов
                chunk = text[start:search_end]
                
                # Пытаемся найти хорошую точку разрыва
                last_space = chunk.rfind(' ', 0, self.chunk_size)
                last_punct = max(
                    chunk.rfind('.', 0, self.chunk_size),
                    chunk.rfind('!', 0, self.chunk_size),
                    chunk.rfind('?', 0, self.chunk_size),
                    chunk.rfind(',', 0, self.chunk_size)
                )
                
                # Выбираем лучшую точку разрыва
                break_point = max(last_punct, last_space)
                
                if break_point > self.chunk_size * 0.7:  # Если нашли разумную точку
                    end = start + break_point + 1
                else:
                    end = start + self.chunk_size
            else:
                end = len(text)
            
            # Извлекаем чанк
            chunk = text[start:end].strip()
            if chunk:  # Добавляем только непустые чанки
                chunks.append(chunk)
            
            # Перемещаемся на следующий чанк с учетом оверлапа
            start = start + self.step_size
            
            # Если осталось меньше чем step_size, берем оставшийся текст
            if start < len(text) and start + self.step_size >= len(text):
                remaining = text[start:].strip()
                if remaining and remaining not in chunks:  # Избегаем дубликатов
                    chunks.append(remaining)
                break
        
        return chunks
    
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
    
    def get_chunk_stats(self, chunks: List[str]) -> dict:
        """
        Получение статистики по чанкам.
        
        Args:
            chunks: Список чанков
        
        Returns:
            Словарь со статистикой
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
                "total_chars": 0
            }
        
        lengths = [len(chunk) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_chars": sum(lengths)
        }

