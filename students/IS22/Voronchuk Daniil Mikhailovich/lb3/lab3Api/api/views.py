from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .embedder import Embedder
from .milvus_client import MilvusClient
import json

class MilvusSearchAPIView(APIView):
 
    embedder = None
    milvus_client = None
    
    @classmethod
    def setup_connections(cls):
        """Инициализация подключений (вызывается один раз)"""
        if cls.embedder is None:
            cls.embedder = Embedder(model_name="intfloat/multilingual-e5-base")
        
        if cls.milvus_client is None:
            cls.milvus_client = MilvusClient(host="localhost", port=19530)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_connections()
    
    def post(self, request, *args, **kwargs):
        query = ""
        
      
        if hasattr(request, 'data') and request.data and isinstance(request.data, dict):
            query = request.data.get("query", "")
        
        # Из request.body
        if not query and hasattr(request, 'body') and request.body:
            try:
                data = json.loads(request.body.decode('utf-8'))
                query = data.get("query", "")
            except:
                pass
        
        if not query:
            return Response(
                {"error": "Запрос не может быть пустым"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Используем уже созданный embedder
            query_embedding = self.embedder.encode_query(query)
            
        except Exception as e:
            return Response(
                {"error": f"Ошибка генерации эмбеддинга: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        collection_name = "documents"
        try:
           
            search_results = self.milvus_client.search(
                collection_name=collection_name,
                query_vectors=[query_embedding],
                top_k=3
            )
            
            formatted_results = []
            
            if search_results and len(search_results) > 0:
                for hit in search_results[0]:
                    formatted_results.append({
                        "distance": round(hit.get("distance", 0), 4), 
                        "text": hit.get("text", "Нет текста"), 
                        "file_name": hit.get("file_name", "N/A"), 
                        "file_path": hit.get("file_path", ""),
                        "chunk_index": hit.get("chunk_index", -1)  
                    })
            
            return Response({
                "success": True,
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": f"Ошибка при выполнении поиска: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )