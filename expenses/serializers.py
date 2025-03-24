from rest_framework import serializers
from .models import Transaction, SubCategory, Category
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email'] # fields you want to expose


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name']

class SubCategorySerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)  # Nested serializer (read-only)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset = Category.objects.all(), source = 'category', write_only = True
    )  # Write only for id

    class Meta:
        model = SubCategory
        fields = ['id', 'name', 'category', 'category_id']



class TransactionSerializer(serializers.ModelSerializer):
    # category = serializers.PrimaryKeyRelatedField(
    #     queryset=Category.objects.all(),
    #     allow_null=True,
    #     required=False
    # )
    # subcategory = serializers.PrimaryKeyRelatedField(
    #     queryset=SubCategory.objects.all(),
    #     allow_null=True,
    #     required=False
    # )

    user = UserSerializer(read_only=True)  # Show user
    user_id = serializers.PrimaryKeyRelatedField(
        queryset = User.objects.all(), source = 'user', write_only = True, required=False
    )

    category = CategorySerializer(read_only=True)  # Nested Serializer
    category_id = serializers.PrimaryKeyRelatedField(
        queryset = Category.objects.all(), source = 'category', allow_null = True, required = False
    )  # Allow setting category by ID

    subcategory = SubCategorySerializer(read_only=True)  # Nested Serializer
    subcategory_id = serializers.PrimaryKeyRelatedField(
       queryset = SubCategory.objects.all(), source = 'subcategory', allow_null = True, required = False
    )  # Allow setting subcategory by ID

    class Meta:
        model = Transaction
        fields = [
            'id',
            'user',
            'user_id',
            'transaction_id',
            'merchant_name',
            'amount',
            'description',
            'time_of_transaction',
            'is_expense',
            'category',
            'category_id',
            'subcategory',
            'subcategory_id',
        ]

    def to_representation(self, instance):
        data = super().to_representation(instance)
        # Add Django user ID to the response
        if 'context' in self and 'django_user_id' in self.context:
            data['django_user_id'] = self.context['django_user_id']
        return data

    def create(self, validated_data):
        # Get the user from the context if not provided in validated_data
        if 'user' not in validated_data and 'request' in self.context:
            validated_data['user'] = self.context['request'].user
        return Transaction.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.transaction_id = validated_data.get('transaction_id', instance.transaction_id)
        instance.merchant_name = validated_data.get('merchant_name', instance.merchant_name)
        instance.amount = validated_data.get('amount', instance.amount)
        instance.description = validated_data.get('description', instance.description)
        instance.time_of_transaction = validated_data.get('time_of_transaction', instance.time_of_transaction)
        instance.is_expense = validated_data.get('is_expense', instance.is_expense)
        instance.category = validated_data.get('category', instance.category)
        instance.subcategory = validated_data.get('subcategory', instance.subcategory)
        instance.user = validated_data.get('user', instance.user)
        instance.save()
        return instance






