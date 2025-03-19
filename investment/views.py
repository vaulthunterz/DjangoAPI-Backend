# investment/views.py
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action

from .services import fetch_market_data, calculate_and_update_disposable_income
from .models import UserProfile, Recommendation, Portfolio, PortfolioItem, FinancialAsset, Alert
from .serializers import UserProfileSerializer, RecommendationSerializer, PortfolioSerializer, PortfolioItemSerializer, \
    FinancialAssetSerializer, AlertSerializer

# Import the new ML system
from ml.recommender.random_forest import RandomForestRecommender
from ml.utils.preprocessing import prepare_user_features, calculate_portfolio_metrics


# --- ViewSets ---

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer

    def get_queryset(self):
        # Ensure users can only access their own profile
        return UserProfile.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # Associate the profile with the current user
        serializer.save(user=self.request.user)

    def perform_update(self, serializer): # Added
      serializer.save(user=self.request.user)


class PortfolioViewSet(viewsets.ModelViewSet):
    queryset = Portfolio.objects.all()
    serializer_class = PortfolioSerializer

    def get_queryset(self):
        return Portfolio.objects.filter(user=self.request.user.userprofile)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user.userprofile)



class PortfolioItemViewSet(viewsets.ModelViewSet):
    queryset = PortfolioItem.objects.all()
    serializer_class = PortfolioItemSerializer

    def get_queryset(self):
        portfolio_id = self.kwargs.get('portfolio_pk')
        if portfolio_id:
            return PortfolioItem.objects.filter(portfolio_id=portfolio_id, portfolio__user=self.request.user.userprofile)
        return PortfolioItem.objects.none()  # Important: Return an empty queryset if no portfolio_id

    def perform_create(self, serializer):
        portfolio_id = self.kwargs.get('portfolio_pk')
        portfolio = get_object_or_404(Portfolio, pk=portfolio_id, user=self.request.user.userprofile)
        serializer.save(portfolio=portfolio)


class FinancialAssetViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = FinancialAsset.objects.all()
    serializer_class = FinancialAssetSerializer
    # Consider adding filtering/searching capabilities here


class RecommendationViewSet(viewsets.ModelViewSet):
    queryset = Recommendation.objects.all()
    serializer_class = RecommendationSerializer
    recommender = RandomForestRecommender()

    def get_queryset(self):
        return Recommendation.objects.filter(user=self.request.user.userprofile)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user.userprofile)

    @action(detail=False, methods=['post'])
    def generate_recommendations(self, request):
        """Generate ML-based investment recommendations"""
        user_profile = request.user.userprofile
        
        # Prepare user features
        user_features = prepare_user_features(user_profile)
        
        # Get ML-based recommendations
        ml_recommendations = self.recommender.predict(user_features)
        
        # Get available assets that match the recommendations
        stored_recommendations = []
        for rec in ml_recommendations:
            matching_assets = FinancialAsset.objects.filter(
                asset_type=rec['asset_type'],
                risk_level__lte=user_profile.risk_tolerance
            )
            
            for asset in matching_assets:
                confidence_msg = f"({rec['confidence']} confidence)"
                recommendation = Recommendation.objects.create(
                    user=user_profile,
                    message=f"Recommended: {asset.name} - {confidence_msg}",
                    financial_asset=asset
                )
                stored_recommendations.append(recommendation)
        
        serializer = self.get_serializer(stored_recommendations, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def create(self, request, *args, **kwargs):
        """Legacy method - redirects to ML-based recommendations"""
        return self.generate_recommendations(request)

    @action(detail=False, methods=['post'])
    def train_model(self, request):
        """Train the ML model with historical data"""
        # Get all user profiles and their portfolio performances
        user_profiles = UserProfile.objects.all()
        portfolio_performances = []
        
        for profile in user_profiles:
            portfolios = Portfolio.objects.filter(user=profile)
            performance = 0
            for portfolio in portfolios:
                metrics = calculate_portfolio_metrics(portfolio)
                performance += metrics['total_value']  # Simplified performance metric
            portfolio_performances.append(performance)
        
        # Prepare and train
        training_data = prepare_training_data(user_profiles, portfolio_performances)
        self.recommender.train(training_data)
        
        return Response({"message": "Model trained successfully"}, status=status.HTTP_200_OK)

class AlertViewSet(viewsets.ModelViewSet):
    queryset = Alert.objects.all()
    serializer_class = AlertSerializer

    def get_queryset(self):
        return Alert.objects.filter(user=self.request.user.userprofile)
    def perform_create(self, serializer):
        serializer.save(user=self.request.user.userprofile)

