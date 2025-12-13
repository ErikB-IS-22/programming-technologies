from rest_framework import serializers

class SearchQuerySerializer(serializers.Serializer):
    query = serializers.CharField()  
    top_k = serializers.IntegerField(default=5)