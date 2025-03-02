from rest_framework import serializers
from .models import Transaction, SubCategory, Category


class SubCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = SubCategory
        fields = ['id', 'name']

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name']

class TransactionSerializer(serializers.ModelSerializer):
    category = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(),
        allow_null=True,
        required=False
    )
    subcategory = serializers.PrimaryKeyRelatedField(
        queryset=SubCategory.objects.all(),
        allow_null=True,
        required=False
    )
    class Meta:
        model = Transaction
        fields = '__all__'

    def update(self, instance, validated_data):
        instance.category = validated_data.get('category', instance.category)
        instance.subcategory = validated_data.get('subcategory', instance.subcategory)
        instance.save()
        return instance



