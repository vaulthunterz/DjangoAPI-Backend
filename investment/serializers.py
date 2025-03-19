# investment/serializers.py
from rest_framework import serializers
from .models import UserProfile, Recommendation, Portfolio, PortfolioItem, FinancialAsset, Alert
from django.contrib.auth.models import User


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']  # Add other relevant user fields


class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)  # Nested user serialization
    user_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(), source='user', write_only=True
    )
    class Meta:
        model = UserProfile
        fields = '__all__'


class FinancialAssetSerializer(serializers.ModelSerializer):
    class Meta:
        model = FinancialAsset
        fields = '__all__'


class PortfolioItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioItem
        fields = '__all__'

class PortfolioSerializer(serializers.ModelSerializer):
    items = PortfolioItemSerializer(many=True, read_only=True)  # Nested items (read-only)
    user = UserProfileSerializer(read_only = True) #Show userprofile
    user_id = serializers.PrimaryKeyRelatedField(
        queryset = UserProfile.objects.all(), source = 'user', write_only = True
    )
    class Meta:
        model = Portfolio
        fields = '__all__'



class RecommendationSerializer(serializers.ModelSerializer):
    user = UserProfileSerializer(read_only = True) #Show userprofile
    user_id = serializers.PrimaryKeyRelatedField(
        queryset = UserProfile.objects.all(), source = 'user', write_only = True
    )
    portfolio = PortfolioSerializer(read_only = True)
    portfolio_id = serializers.PrimaryKeyRelatedField(
        queryset = Portfolio.objects.all(), source = 'portfolio', allow_null = True, required = False
    )
    financial_asset = FinancialAssetSerializer(read_only = True)
    financial_asset_id = serializers.PrimaryKeyRelatedField(
      queryset = FinancialAsset.objects.all(), source = 'financial_asset', allow_null = True, required = False
    )

    class Meta:
        model = Recommendation
        fields = '__all__'


class AlertSerializer(serializers.ModelSerializer):
    user = UserProfileSerializer(read_only=True)
    user_id = serializers.PrimaryKeyRelatedField(
        queryset = UserProfile.objects.all(), source = 'user', write_only = True
    )
    portfolio_item = PortfolioItemSerializer(read_only = True)
    portfolio_item_id = serializers.PrimaryKeyRelatedField(
        queryset = PortfolioItem.objects.all(), source = 'portfolio_item', allow_null = True, required = False
    )
    class Meta:
        model = Alert
        fields = '__all__'