# investment/views.py
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.views import APIView
from django.utils import timezone
import json

from .services import fetch_market_data, calculate_and_update_disposable_income
from .models import UserProfile, Recommendation, Portfolio, PortfolioItem, MoneyMarketFund, Alert, InvestmentQuestionnaire
from .serializers import UserProfileSerializer, RecommendationSerializer, PortfolioSerializer, PortfolioItemSerializer, \
    MoneyMarketFundSerializer, AlertSerializer, InvestmentQuestionnaireSerializer

# Import the ML system
from ml.recommender.random_forest import RandomForestRecommender
from ml.recommender.hybrid_recommender import HybridRecommender
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


class InvestmentQuestionnaireViewSet(viewsets.ModelViewSet):
    queryset = InvestmentQuestionnaire.objects.all()
    serializer_class = InvestmentQuestionnaireSerializer
    
    def get_queryset(self):
        # Ensure users can only access their own questionnaire
        return InvestmentQuestionnaire.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        # Associate the questionnaire with the current user
        serializer.save(user=self.request.user)
        
        # Update user profile with questionnaire data
        user_profile = UserProfile.objects.get(user=self.request.user)
        questionnaire = serializer.instance
        
        # Update risk tolerance based on questionnaire
        if questionnaire.risk_tolerance_score:
            # Map 1-10 score to 1-3 scale
            risk_tolerance = min(3, max(1, (questionnaire.risk_tolerance_score // 4) + 1))
            user_profile.risk_tolerance = risk_tolerance
        
        # Update investment experience based on knowledge
        if questionnaire.investment_knowledge:
            knowledge_to_experience = {
                'none': 'beginner',
                'limited': 'beginner',
                'good': 'intermediate',
                'extensive': 'advanced'
            }
            user_profile.investment_experience = knowledge_to_experience.get(
                questionnaire.investment_knowledge, 
                user_profile.investment_experience
            )
        
        # Update investment timeline based on years to goal
        if questionnaire.years_to_goal:
            if questionnaire.years_to_goal < 3:
                user_profile.investment_timeline = 'short'
            elif questionnaire.years_to_goal < 10:
                user_profile.investment_timeline = 'mid'
            else:
                user_profile.investment_timeline = 'long'
        
        # Update investment goals
        if questionnaire.primary_goal:
            user_profile.investment_goals = questionnaire.primary_goal
        
        # Update investment preferences
        if questionnaire.preferred_asset_types:
            user_profile.investment_preference = questionnaire.preferred_asset_types
        
        # Save updated profile
        user_profile.save()
        
        # Generate recommendations based on questionnaire
        self._generate_recommendations(user_profile, questionnaire)

    @action(detail=False, methods=['get'])
    def status(self, request):
        """Check if the user has completed the questionnaire"""
        try:
            questionnaire = InvestmentQuestionnaire.objects.get(user=request.user)
            return Response({
                'isCompleted': True,
                'data': self.get_serializer(questionnaire).data
            })
        except InvestmentQuestionnaire.DoesNotExist:
            return Response({
                'isCompleted': False,
                'data': None
            })

    def _generate_recommendations(self, user_profile, questionnaire):
        """Generate recommendations based on questionnaire data"""
        # Initialize hybrid recommender
        recommender = HybridRecommender()
        
        # Prepare user features
        user_features = {
            'user_profile': user_profile,
            'questionnaire': questionnaire
        }
        
        # Get recommendations
        recommendations = recommender.predict(user_features)
        
        # Store top 5 recommendations
        for i, rec in enumerate(recommendations[:5]):
            # Create a money market fund record if it doesn't exist
            fund_name = rec['fund_name']
            fund_info = rec['fund_info']
            
            # Try to find an existing money market fund with this name
            try:
                money_market_fund = MoneyMarketFund.objects.get(name=fund_name)
            except MoneyMarketFund.DoesNotExist:
                # Create a new money market fund
                money_market_fund = MoneyMarketFund.objects.create(
                    name=fund_name,
                    symbol=f"MMF-{i+1}",  # Generate a symbol
                    description=fund_info['description'],
                    fund_manager=fund_name.split(' ')[0],  # Extract fund manager name
                    risk_level=fund_info['risk_level'],
                    min_investment=fund_info['min_investment'],
                    expected_returns=fund_info['returns'],
                    liquidity='High',  # Money market funds are highly liquid
                    fees='0.5-1.5% annually'  # Typical fee range
                )
            
            Recommendation.objects.create(
                user=user_profile,
                message=f"Recommended: {fund_name}",
                financial_asset=money_market_fund,
                confidence_score=rec['score'],
                recommendation_type=rec['recommendation_type'],
                explanation=rec['explanation']
            )


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


class MoneyMarketFundViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = MoneyMarketFund.objects.all()
    serializer_class = MoneyMarketFundSerializer
    # Consider adding filtering/searching capabilities here


class RecommendationViewSet(viewsets.ModelViewSet):
    queryset = Recommendation.objects.all()
    serializer_class = RecommendationSerializer
    recommender = RandomForestRecommender()
    hybrid_recommender = HybridRecommender()

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
        
        # Get available money market funds that match the recommendations
        stored_recommendations = []
        for rec in ml_recommendations:
            matching_funds = MoneyMarketFund.objects.filter(
                risk_level__lte=user_profile.risk_tolerance
            )
            
            for fund in matching_funds:
                confidence_msg = f"({rec['confidence']} confidence)"
                recommendation = Recommendation.objects.create(
                    user=user_profile,
                    message=f"Recommended: {fund.name} - {confidence_msg}",
                    financial_asset=fund
                )
                stored_recommendations.append(recommendation)
        
        serializer = self.get_serializer(stored_recommendations, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def create(self, request, *args, **kwargs):
        """Legacy method - redirects to ML-based recommendations"""
        return self.generate_recommendations(request)

    @action(detail=False, methods=['post'])
    def generate_hybrid_recommendations(self, request):
        """Generate hybrid recommendations using both rule-based and expense-based approaches"""
        user_profile = request.user.userprofile
        
        # Get questionnaire if available
        try:
            questionnaire = InvestmentQuestionnaire.objects.get(user=request.user)
        except InvestmentQuestionnaire.DoesNotExist:
            questionnaire = None
        
        # Prepare user features
        user_features = {
            'user_profile': user_profile,
            'questionnaire': questionnaire
        }
        
        # Get hybrid recommendations
        recommendations = self.hybrid_recommender.predict(user_features)
        
        # Store recommendations
        stored_recommendations = []
        for rec in recommendations[:5]:  # Store top 5 recommendations
            # Create a money market fund record if it doesn't exist
            fund_name = rec['fund_name']
            fund_info = rec['fund_info']
            
            # Try to find an existing money market fund with this name
            try:
                money_market_fund = MoneyMarketFund.objects.get(name=fund_name)
            except MoneyMarketFund.DoesNotExist:
                # Create a new money market fund
                money_market_fund = MoneyMarketFund.objects.create(
                    name=fund_name,
                    symbol=f"MMF-{len(stored_recommendations)+1}",  # Generate a symbol
                    description=fund_info['description'],
                    fund_manager=fund_name.split(' ')[0],  # Extract fund manager name
                    risk_level=fund_info['risk_level'],
                    min_investment=fund_info['min_investment'],
                    expected_returns=fund_info['returns'],
                    liquidity='High',  # Money market funds are highly liquid
                    fees='0.5-1.5% annually'  # Typical fee range
                )
            
            recommendation = Recommendation.objects.create(
                user=user_profile,
                message=f"Recommended: {fund_name}",
                financial_asset=money_market_fund,
                confidence_score=rec['score'],
                recommendation_type=rec['recommendation_type'],
                explanation=rec['explanation']
            )
            stored_recommendations.append(recommendation)
        
        serializer = self.get_serializer(stored_recommendations, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

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


class ExpenseBasedRecommendationView(APIView):
    """API view for generating recommendations based on expense patterns"""
    
    def post(self, request, *args, **kwargs):
        user_profile = request.user.userprofile
        
        # Initialize hybrid recommender
        recommender = HybridRecommender()
        
        # Prepare user features
        user_features = {
            'user_profile': user_profile,
            'questionnaire': None
        }
        
        # Get expense-based recommendations
        recommendations = recommender._get_expense_based_recommendations(user_profile)
        
        # Store recommendations
        stored_recommendations = []
        for rec in recommendations[:5]:  # Store top 5 recommendations
            # Create a money market fund record if it doesn't exist
            fund_name = rec['fund_name']
            fund_info = rec['fund_info']
            
            # Try to find an existing money market fund with this name
            try:
                money_market_fund = MoneyMarketFund.objects.get(name=fund_name)
            except MoneyMarketFund.DoesNotExist:
                # Create a new money market fund
                money_market_fund = MoneyMarketFund.objects.create(
                    name=fund_name,
                    symbol=f"MMF-{len(stored_recommendations)+1}",  # Generate a symbol
                    description=fund_info['description'],
                    fund_manager=fund_name.split(' ')[0],  # Extract fund manager name
                    risk_level=fund_info['risk_level'],
                    min_investment=fund_info['min_investment'],
                    expected_returns=fund_info['returns'],
                    liquidity='High',  # Money market funds are highly liquid
                    fees='0.5-1.5% annually'  # Typical fee range
                )
            
            recommendation = Recommendation.objects.create(
                user=user_profile,
                message=f"Recommended: {fund_name}",
                financial_asset=money_market_fund,
                confidence_score=rec['score'],
                recommendation_type='expense_based',
                explanation=rec['explanation']
            )
            stored_recommendations.append(recommendation)
        
        serializer = RecommendationSerializer(stored_recommendations, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

