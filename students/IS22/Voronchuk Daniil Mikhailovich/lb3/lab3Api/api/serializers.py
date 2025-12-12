from rest_framework import serializers

class SearchSerializer(serializers.Serializer):
    query_vectors = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    top_k = serializers.IntegerField(default=5)
    expr = serializers.CharField(required=False, allow_blank=True)