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
from .ml.recommender.random_forest import RandomForestRecommender
from .ml.recommender.hybrid_recommender import HybridRecommender
from .ml.recommender.advanced_hybrid_recommender import AdvancedHybridRecommender
from .ml.utils.preprocessing import prepare_user_features, calculate_portfolio_metrics

# Import pagination
from .pagination import StandardResultsSetPagination, PortfolioPagination, RecommendationPagination


# --- ViewSets ---

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    pagination_class = StandardResultsSetPagination

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
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        # Ensure users can only access their own questionnaire
        return InvestmentQuestionnaire.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # Check if a questionnaire already exists for this user
        try:
            existing_questionnaire = InvestmentQuestionnaire.objects.get(user=self.request.user)
            # Update the existing questionnaire instead of creating a new one
            for key, value in serializer.validated_data.items():
                setattr(existing_questionnaire, key, value)
            existing_questionnaire.save()
            questionnaire = existing_questionnaire
        except InvestmentQuestionnaire.DoesNotExist:
            # Create a new questionnaire
            questionnaire = serializer.save(user=self.request.user)

        # Update user profile with questionnaire data
        try:
            user_profile = UserProfile.objects.get(user=self.request.user)
        except UserProfile.DoesNotExist:
            # Create a new user profile with default values
            user_profile = UserProfile.objects.create(
                user=self.request.user,
                risk_tolerance=1,  # Default to low risk tolerance
                investment_experience='beginner',  # Default to beginner
                investment_timeline='mid',  # Default to mid-term
                investment_goals='General Investing'  # Default goal
            )

        try:
            # Update risk tolerance based on questionnaire's investment_preference
            if questionnaire.investment_preference:
                # Map investment_preference to risk_tolerance (1-3 scale)
                preference_to_risk = {
                    'very_safe': 1,      # Low risk
                    'conservative': 1,   # Low risk
                    'balanced': 2,       # Medium risk
                    'growth': 3,         # High risk
                    'aggressive': 3      # High risk
                }
                user_profile.risk_tolerance = preference_to_risk.get(
                    questionnaire.investment_preference,
                    1  # Default to low risk if not found
                )

            # Update investment experience based on knowledge
            if questionnaire.investment_knowledge:
                knowledge_to_experience = {
                    'none': 'beginner',
                    'basic': 'beginner',
                    'moderate': 'intermediate',
                    'good': 'intermediate',
                    'expert': 'advanced'
                }
                user_profile.investment_experience = knowledge_to_experience.get(
                    questionnaire.investment_knowledge,
                    'beginner'  # Default to beginner if not found
                )

            # Update investment timeline based on investment_timeframe
            if questionnaire.investment_timeframe:
                timeframe_to_timeline = {
                    'very_short': 'short',  # Less than 1 year
                    'short': 'short',       # 1-3 years
                    'medium': 'mid',        # 3-5 years
                    'long': 'mid',          # 5-10 years
                    'very_long': 'long'     # More than 10 years
                }
                user_profile.investment_timeline = timeframe_to_timeline.get(
                    questionnaire.investment_timeframe,
                    'mid'  # Default to mid-term if not found
                )

            # Update investment goals
            if questionnaire.primary_goal:
                user_profile.investment_goals = questionnaire.primary_goal

            # Save the user profile
            user_profile.save()
        except Exception as e:
            # Log the error but don't fail the questionnaire submission
            print(f"Error updating user profile: {str(e)}")

        # Generate recommendations based on questionnaire
        self._generate_recommendations(user_profile, questionnaire)

        # Return the questionnaire instance for the response
        return questionnaire

    @action(detail=False, methods=['get'])
    def status(self, request):
        """Check if the user has completed the questionnaire and return analytics"""
        try:
            questionnaire = InvestmentQuestionnaire.objects.get(user=request.user)

            # Get user profile for additional analytics
            try:
                user_profile = UserProfile.objects.get(user=request.user)
                profile_data = UserProfileSerializer(user_profile).data
            except UserProfile.DoesNotExist:
                profile_data = None

            # Calculate analytics based on questionnaire
            analytics = self._calculate_analytics(questionnaire, user_profile if 'user_profile' in locals() else None)

            return Response({
                'isCompleted': True,
                'data': self.get_serializer(questionnaire).data,
                'profile': profile_data,
                'analytics': analytics
            })
        except InvestmentQuestionnaire.DoesNotExist:
            return Response({
                'isCompleted': False,
                'data': None,
                'profile': None,
                'analytics': None
            })

    def _calculate_analytics(self, questionnaire, user_profile=None):
        """Calculate analytics based on questionnaire responses"""
        analytics = {}

        # Risk score calculation (1-10 scale)
        risk_score = 5  # Default medium risk

        # Adjust based on investment preference
        if questionnaire.investment_preference:
            preference_to_risk = {
                'very_safe': 1,
                'conservative': 3,
                'balanced': 5,
                'growth': 7,
                'aggressive': 10
            }
            risk_score = preference_to_risk.get(questionnaire.investment_preference, risk_score)

        # Adjust based on market drop reaction
        if questionnaire.market_drop_reaction:
            reaction_adjustment = {
                'sell_all': -2,
                'sell_some': -1,
                'do_nothing': 0,
                'buy_more': 2,
                'seek_advice': 0
            }
            risk_score += reaction_adjustment.get(questionnaire.market_drop_reaction, 0)

        # Adjust based on loss tolerance
        if questionnaire.loss_tolerance:
            loss_adjustment = {
                '0-5': -2,
                '5-10': -1,
                '10-20': 0,
                '20-30': 1,
                '30+': 2
            }
            risk_score += loss_adjustment.get(questionnaire.loss_tolerance, 0)

        # Ensure risk score is within 1-10 range
        risk_score = max(1, min(10, risk_score))

        # Map 1-10 risk score to 1-3 scale for user profile
        profile_risk_level = 1 if risk_score <= 3 else (2 if risk_score <= 7 else 3)

        # Calculate recommended allocation
        stocks_percent = 20 + (risk_score * 5)  # 25% to 70%
        bonds_percent = 70 - (risk_score * 5)   # 65% to 20%
        cash_percent = 100 - stocks_percent - bonds_percent

        # Adjust based on timeline
        if questionnaire.investment_timeframe:
            if questionnaire.investment_timeframe in ['long', 'very_long']:
                stocks_percent += 10
                bonds_percent -= 5
                cash_percent -= 5
            elif questionnaire.investment_timeframe in ['very_short', 'short']:
                stocks_percent -= 10
                bonds_percent += 5
                cash_percent += 5

        # Ensure percentages are within reasonable ranges and sum to 100%
        stocks_percent = max(0, min(100, stocks_percent))
        bonds_percent = max(0, min(100, bonds_percent))
        cash_percent = max(0, min(100, 100 - stocks_percent - bonds_percent))

        # Store analytics
        analytics['risk_score'] = risk_score
        analytics['profile_risk_level'] = profile_risk_level
        analytics['allocation'] = {
            'stocks': round(stocks_percent),
            'bonds': round(bonds_percent),
            'cash': round(cash_percent)
        }

        # Include user's preferred investment types
        if hasattr(questionnaire, 'preferred_investment_types') and questionnaire.preferred_investment_types:
            try:
                if isinstance(questionnaire.preferred_investment_types, str):
                    # Try to parse JSON string
                    import json
                    preferred_types = json.loads(questionnaire.preferred_investment_types)
                else:
                    preferred_types = questionnaire.preferred_investment_types

                if preferred_types and isinstance(preferred_types, list):
                    analytics['preferred_investment_types'] = preferred_types
            except Exception as e:
                print(f"Error processing preferred investment types: {str(e)}")

        # Include ethical preferences if available
        if hasattr(questionnaire, 'ethical_preferences') and questionnaire.ethical_preferences:
            try:
                if isinstance(questionnaire.ethical_preferences, str):
                    import json
                    ethical_prefs = json.loads(questionnaire.ethical_preferences)
                else:
                    ethical_prefs = questionnaire.ethical_preferences

                if ethical_prefs and isinstance(ethical_prefs, list):
                    analytics['ethical_preferences'] = ethical_prefs
            except Exception as e:
                print(f"Error processing ethical preferences: {str(e)}")

        # Include sector preferences if available
        if hasattr(questionnaire, 'sector_preferences') and questionnaire.sector_preferences:
            try:
                if isinstance(questionnaire.sector_preferences, str):
                    import json
                    sector_prefs = json.loads(questionnaire.sector_preferences)
                else:
                    sector_prefs = questionnaire.sector_preferences

                if sector_prefs and isinstance(sector_prefs, list):
                    analytics['sector_preferences'] = sector_prefs
            except Exception as e:
                print(f"Error processing sector preferences: {str(e)}")

        # Investment style based on risk and experience
        if user_profile:
            experience = user_profile.investment_experience
            if profile_risk_level == 1:
                analytics['investment_style'] = 'Conservative' if experience == 'beginner' else 'Income-Focused'
            elif profile_risk_level == 2:
                analytics['investment_style'] = 'Balanced' if experience == 'beginner' else 'Growth & Income'
            else:
                analytics['investment_style'] = 'Growth' if experience != 'advanced' else 'Aggressive Growth'
        else:
            # Default based just on risk
            styles = ['Conservative', 'Balanced', 'Growth']
            analytics['investment_style'] = styles[profile_risk_level - 1]

        return analytics

    def _generate_recommendations(self, user_profile, questionnaire):
        """Generate recommendations based on questionnaire data"""
        try:
            # Initialize advanced hybrid recommender with all the enhanced features
            recommender = AdvancedHybridRecommender()

            # Prepare user features
            user_features = {
                'user_profile': user_profile,
                'questionnaire': questionnaire
            }

            # Get recommendations with advanced features
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
        except Exception as e:
            # Log the error but don't fail the questionnaire submission
            print(f"Error generating recommendations: {str(e)}")
            # We could also create a generic recommendation here if needed


class PortfolioViewSet(viewsets.ModelViewSet):
    queryset = Portfolio.objects.all()
    serializer_class = PortfolioSerializer
    pagination_class = PortfolioPagination

    def get_queryset(self):
        return Portfolio.objects.filter(user=self.request.user.userprofile)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user.userprofile)



class PortfolioItemViewSet(viewsets.ModelViewSet):
    queryset = PortfolioItem.objects.all()
    serializer_class = PortfolioItemSerializer
    pagination_class = StandardResultsSetPagination

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
    pagination_class = StandardResultsSetPagination
    # Consider adding filtering/searching capabilities here


class RecommendationViewSet(viewsets.ModelViewSet):
    queryset = Recommendation.objects.all()
    serializer_class = RecommendationSerializer
    pagination_class = RecommendationPagination
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
        """Generate advanced hybrid recommendations with all enhanced features"""
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

        # Initialize advanced hybrid recommender
        advanced_recommender = AdvancedHybridRecommender()

        # Get advanced hybrid recommendations with all enhanced features
        recommendations = advanced_recommender.predict(user_features)

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

        # Initialize advanced recommender
        advanced_recommender = AdvancedHybridRecommender()

        # Prepare training data
        from .ml.utils.preprocessing import prepare_training_data
        training_data = prepare_training_data(user_profiles, portfolio_performances)

        # Train the model
        advanced_recommender.train(training_data)

        return Response({"message": "Model trained successfully"}, status=status.HTTP_200_OK)

class AlertViewSet(viewsets.ModelViewSet):
    queryset = Alert.objects.all()
    serializer_class = AlertSerializer
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        return Alert.objects.filter(user=self.request.user.userprofile)
    def perform_create(self, serializer):
        serializer.save(user=self.request.user.userprofile)


class ExpenseBasedRecommendationView(APIView):
    """API view for generating recommendations based on expense patterns"""

    def post(self, request, *args, **kwargs):
        user_profile = request.user.userprofile

        # Initialize advanced hybrid recommender
        recommender = AdvancedHybridRecommender()

        # Prepare user features
        user_features = {
            'user_profile': user_profile,
            'questionnaire': None
        }

        # Get expense-based recommendations using the advanced recommender
        # First get the full recommendations
        recommendations = recommender.predict(user_features)

        # Then filter to only include expense-based ones
        recommendations = [rec for rec in recommendations if rec.get('recommendation_type') == 'expense_based']

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


class PortfolioSummaryView(APIView):
    """API view for getting a summary of all user portfolios"""

    def get(self, request, *args, **kwargs):
        user_profile = request.user.userprofile

        # Get all portfolios for the user
        portfolios = Portfolio.objects.filter(user=user_profile)

        if not portfolios.exists():
            return Response({
                'total_invested': 0,
                'current_value': 0,
                'returns': 0,
                'returns_percentage': 0,
                'asset_allocation': []
            })

        # Calculate total invested amount
        total_invested = sum(float(portfolio.total_amount) for portfolio in portfolios)

        # Get all portfolio items
        portfolio_items = PortfolioItem.objects.filter(portfolio__in=portfolios)

        # Calculate current value (for now, we'll use the same as invested since we don't have real-time data)
        # In a real app, you would fetch current market prices for each asset
        current_value = sum(float(item.quantity * item.buy_price) for item in portfolio_items)

        # If no items, use the portfolio total_amount as current value
        if not portfolio_items.exists():
            current_value = total_invested

        # Calculate returns
        returns = current_value - total_invested
        returns_percentage = (returns / total_invested * 100) if total_invested > 0 else 0

        # Calculate asset allocation
        asset_allocation = {}
        for item in portfolio_items:
            asset_type = item.asset_name.split(' ')[0]  # Simplified: use first word as asset type
            item_value = float(item.quantity * item.buy_price)

            if asset_type in asset_allocation:
                asset_allocation[asset_type] += item_value
            else:
                asset_allocation[asset_type] = item_value

        # Convert to percentage and format for frontend
        asset_allocation_list = []
        for asset_type, value in asset_allocation.items():
            percentage = (value / current_value * 100) if current_value > 0 else 0
            asset_allocation_list.append({
                'type': asset_type,
                'percentage': round(percentage, 2),
                'value': round(value, 2)
            })

        # Sort by value (descending)
        asset_allocation_list.sort(key=lambda x: x['value'], reverse=True)

        return Response({
            'total_invested': round(total_invested, 2),
            'current_value': round(current_value, 2),
            'returns': round(returns, 2),
            'returns_percentage': round(returns_percentage, 2),
            'asset_allocation': asset_allocation_list
        })

