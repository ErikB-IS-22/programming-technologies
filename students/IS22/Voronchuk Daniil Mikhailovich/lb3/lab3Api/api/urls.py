from django.urls import path
from .views import MilvusSearchAPIView

urlpatterns = [
    path('milvus/search/', MilvusSearchAPIView.as_view(), name='milvus-search'),
]