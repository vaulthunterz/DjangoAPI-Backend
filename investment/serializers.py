# investment/serializers.py
from rest_framework import serializers
from .models import UserProfile, Recommendation, Portfolio, PortfolioItem, MoneyMarketFund, Alert, InvestmentQuestionnaire
from django.contrib.auth.models import User


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']  # Add other relevant user fields
        ref_name = 'InvestmentUserSerializer'


class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)  # Nested user serialization
    user_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(), source='user', write_only=True
    )
    class Meta:
        model = UserProfile
        fields = '__all__'
        read_only_fields = ('user',)


class MoneyMarketFundSerializer(serializers.ModelSerializer):
    class Meta:
        model = MoneyMarketFund
        fields = '__all__'


class PortfolioItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioItem
        fields = '__all__'
        read_only_fields = ('portfolio',)


class PortfolioSerializer(serializers.ModelSerializer):
    items = PortfolioItemSerializer(many=True, read_only=True)  # Nested items (read-only)
    user = UserProfileSerializer(read_only = True) #Show userprofile
    user_id = serializers.PrimaryKeyRelatedField(
        queryset = UserProfile.objects.all(), source = 'user', write_only = True
    )
    class Meta:
        model = Portfolio
        fields = '__all__'
        read_only_fields = ('user',)


class RecommendationSerializer(serializers.ModelSerializer):
    user = UserProfileSerializer(read_only = True) #Show userprofile
    user_id = serializers.PrimaryKeyRelatedField(
        queryset = UserProfile.objects.all(), source = 'user', write_only = True
    )
    portfolio = PortfolioSerializer(read_only = True)
    portfolio_id = serializers.PrimaryKeyRelatedField(
        queryset = Portfolio.objects.all(), source = 'portfolio', allow_null = True, required = False
    )
    financial_asset = MoneyMarketFundSerializer(read_only = True)
    financial_asset_id = serializers.PrimaryKeyRelatedField(
      queryset = MoneyMarketFund.objects.all(), source = 'financial_asset', allow_null = True, required = False
    )

    class Meta:
        model = Recommendation
        fields = '__all__'
        read_only_fields = ('user',)


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
        read_only_fields = ('user',)


class InvestmentQuestionnaireSerializer(serializers.ModelSerializer):
    class Meta:
        model = InvestmentQuestionnaire
        fields = '__all__'
        read_only_fields = ('user', 'expense_categories', 'income_sources')