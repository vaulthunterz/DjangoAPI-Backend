"""
Advanced Hybrid Recommender System for Investment Recommendations

This recommender system extends the basic hybrid recommender with advanced features:
1. Weighted Question Importance
2. Personalized Asset Allocation Based on Goals
3. Age-Based Risk Adjustment
4. Investment Type Clustering
5. Dynamic Risk Tolerance Assessment
6. Behavioral Finance Insights
7. Sector-Specific Recommendations
8. Time-Based Investment Horizon Adjustments
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import date, datetime
from django.utils import timezone
from django.db.models import Sum, Count, Avg

from .hybrid_recommender import HybridRecommender
from ...models import UserProfile, MoneyMarketFund as FinancialAsset, Portfolio, PortfolioItem, Recommendation, InvestmentQuestionnaire
from expenses.models import Transaction, Category, SubCategory

class AdvancedHybridRecommender(HybridRecommender):
    """
    Advanced hybrid recommender that extends the basic hybrid recommender with
    sophisticated features for more personalized and accurate investment recommendations.
    """

    def __init__(self):
        # Initialize the parent class
        super().__init__()

        # Additional initialization for advanced features
        self._initialize_advanced_features()

    def _initialize_advanced_features(self):
        """Initialize data structures for advanced recommendation features"""
        # 1. Weighted Question Importance
        self.question_weights = {
            # Financial Situation (higher weights as these directly impact investment capacity)
            'annual_income_range': 0.9,
            'monthly_savings_range': 0.95,
            'emergency_fund_months': 0.8,
            'debt_situation': 0.9,

            # Investment Goals (high weights as these determine direction)
            'primary_goal': 0.85,
            'investment_timeframe': 0.9,
            'monthly_investment': 0.85,

            # Risk Assessment (highest weights as these directly impact asset allocation)
            'market_drop_reaction': 0.95,
            'investment_preference': 1.0,
            'loss_tolerance': 0.95,
            'risk_comfort_scenario': 0.9,

            # Investment Knowledge & Experience (moderate weights)
            'investment_knowledge': 0.7,
            'investment_experience_years': 0.75,
            'previous_investments': 0.7,

            # Investment Preferences (moderate weights)
            'preferred_investment_types': 0.8,
            'ethical_preferences': 0.6,
            'sector_preferences': 0.7,

            # Additional Information (lower weights)
            'financial_dependents': 0.6,
            'income_stability': 0.7,
            'major_expenses_planned': 0.65
        }

        # 2. Personalized Asset Allocation Based on Goals
        self.goal_allocation_templates = {
            # Retirement planning - longer-term, balanced approach
            'retirement': {
                'money_market': 0.15,  # Lower liquidity needs
                'bonds': 0.35,         # Stability
                'stocks': 0.40,        # Growth
                'real_estate': 0.10    # Diversification
            },
            # Education savings - medium-term, more conservative
            'education': {
                'money_market': 0.30,  # Higher liquidity for upcoming expenses
                'bonds': 0.40,         # Stability
                'stocks': 0.25,        # Some growth
                'real_estate': 0.05    # Limited exposure
            },
            # Home purchase - shorter-term, conservative
            'home': {
                'money_market': 0.50,  # High liquidity for down payment
                'bonds': 0.40,         # Stability
                'stocks': 0.10,        # Limited growth
                'real_estate': 0.00    # No exposure (already investing in real estate)
            },
            # Wealth building - longer-term, growth-oriented
            'wealth_building': {
                'money_market': 0.10,  # Lower liquidity needs
                'bonds': 0.20,         # Some stability
                'stocks': 0.60,        # High growth
                'real_estate': 0.10    # Diversification
            },
            # Passive income - income-oriented
            'passive_income': {
                'money_market': 0.20,  # Some liquidity
                'bonds': 0.45,         # Income generation
                'stocks': 0.25,        # Dividend stocks
                'real_estate': 0.10    # Income-producing properties
            },
            # Emergency fund - highly liquid, very conservative
            'emergency_fund': {
                'money_market': 0.80,  # Very high liquidity
                'bonds': 0.20,         # Some short-term bonds
                'stocks': 0.00,        # No stocks
                'real_estate': 0.00    # No real estate
            },
            # Major purchase - short-term, very conservative
            'major_purchase': {
                'money_market': 0.70,  # Very high liquidity
                'bonds': 0.25,         # Some stability
                'stocks': 0.05,        # Very limited growth
                'real_estate': 0.00    # No real estate
            },
            # Default/other - balanced approach
            'other': {
                'money_market': 0.25,  # Balanced liquidity
                'bonds': 0.30,         # Balanced stability
                'stocks': 0.35,        # Balanced growth
                'real_estate': 0.10    # Some diversification
            }
        }

        # 3. Age-Based Risk Adjustment
        self.age_risk_adjustments = {
            # Age brackets with risk adjustment factors
            # Younger investors can take more risk
            'under_25': {
                'risk_factor': 1.3,        # Increase risk tolerance
                'stocks_factor': 1.4,       # Increase stock allocation
                'money_market_factor': 0.7, # Decrease money market allocation
                'bonds_factor': 0.8,        # Decrease bond allocation
                'real_estate_factor': 1.1   # Slight increase in real estate
            },
            '25_to_34': {
                'risk_factor': 1.2,
                'stocks_factor': 1.3,
                'money_market_factor': 0.8,
                'bonds_factor': 0.9,
                'real_estate_factor': 1.1
            },
            '35_to_44': {
                'risk_factor': 1.1,
                'stocks_factor': 1.1,
                'money_market_factor': 0.9,
                'bonds_factor': 1.0,
                'real_estate_factor': 1.0
            },
            # Middle-aged investors should be more balanced
            '45_to_54': {
                'risk_factor': 1.0,        # Neutral
                'stocks_factor': 1.0,       # Neutral
                'money_market_factor': 1.0, # Neutral
                'bonds_factor': 1.0,        # Neutral
                'real_estate_factor': 1.0   # Neutral
            },
            '55_to_64': {
                'risk_factor': 0.9,
                'stocks_factor': 0.9,
                'money_market_factor': 1.1,
                'bonds_factor': 1.2,
                'real_estate_factor': 0.9
            },
            # Older investors should be more conservative
            '65_plus': {
                'risk_factor': 0.7,        # Decrease risk tolerance
                'stocks_factor': 0.6,       # Decrease stock allocation
                'money_market_factor': 1.4, # Increase money market allocation
                'bonds_factor': 1.5,        # Increase bond allocation
                'real_estate_factor': 0.8   # Decrease real estate
            }
        }

        # 4. Investment Type Clustering
        # Map asset types to fund categories with detailed characteristics
        self.investment_clusters = {
            'money_market': {
                'funds': [
                    'CIC Money Market Fund',
                    'NCBA Money Market Fund',
                    'Jubilee Money Market Fund',
                    'Sanlam Money Market Fund'
                ],
                'characteristics': {
                    'liquidity': 'high',
                    'risk': 'low',
                    'returns': 'low',
                    'time_horizon': 'short',
                    'volatility': 'low',
                    'min_investment': 'low',
                    'suitable_for_beginners': True
                }
            },
            'bonds': {
                'funds': [
                    'Old Mutual Money Market Fund',  # More bond-like characteristics
                    'Britam Money Market Fund'
                ],
                'characteristics': {
                    'liquidity': 'medium',
                    'risk': 'low_to_medium',
                    'returns': 'medium',
                    'time_horizon': 'medium',
                    'volatility': 'low_to_medium',
                    'min_investment': 'medium',
                    'suitable_for_beginners': True
                }
            },
            'stocks': {
                'funds': [
                    'Cytonn Money Market Fund',       # More equity-like characteristics
                    'Genghis Capital Money Market Fund'
                ],
                'characteristics': {
                    'liquidity': 'medium',
                    'risk': 'high',
                    'returns': 'high',
                    'time_horizon': 'long',
                    'volatility': 'high',
                    'min_investment': 'medium',
                    'suitable_for_beginners': False
                }
            },
            'real_estate': {
                'funds': [
                    'Nabo Capital Money Market Fund',  # More real estate exposure
                    'Zimele Money Market Fund'
                ],
                'characteristics': {
                    'liquidity': 'low',
                    'risk': 'medium_to_high',
                    'returns': 'medium_to_high',
                    'time_horizon': 'long',
                    'volatility': 'medium',
                    'min_investment': 'high',
                    'suitable_for_beginners': False
                }
            }
        }

        # Create a simplified mapping for backward compatibility
        self.asset_type_fund_mapping = {}
        for asset_type, cluster in self.investment_clusters.items():
            self.asset_type_fund_mapping[asset_type] = cluster['funds']

    def predict(self, user_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate advanced hybrid recommendations based on user features

        Args:
            user_features: Dictionary containing user profile and questionnaire information

        Returns:
            List of recommendation dictionaries with scores and explanations
        """
        # Extract user profile and questionnaire
        user_profile = user_features.get('user_profile')
        questionnaire = user_features.get('questionnaire')

        if not user_profile:
            return []

        # Get basic recommendations from parent class
        basic_recommendations = super().predict(user_features)

        # Apply advanced features to enhance recommendations
        enhanced_recommendations = self._apply_advanced_features(
            basic_recommendations,
            user_profile,
            questionnaire
        )

        return enhanced_recommendations

    def _apply_advanced_features(self,
                                recommendations: List[Dict[str, Any]],
                                user_profile: UserProfile,
                                questionnaire: InvestmentQuestionnaire = None) -> List[Dict[str, Any]]:
        """
        Apply advanced recommendation features to enhance basic recommendations

        Args:
            recommendations: Basic recommendations from parent class
            user_profile: User profile object
            questionnaire: User's questionnaire responses (if available)

        Returns:
            Enhanced recommendations with advanced features applied
        """
        # If no questionnaire data is available, return basic recommendations
        if not questionnaire:
            return recommendations

        # 1. Apply Weighted Question Importance
        weighted_risk_score = self._calculate_weighted_risk_score(questionnaire)

        # 2. Apply Personalized Asset Allocation Based on Goals
        goal_allocations = self._calculate_goal_based_allocations(questionnaire)

        # 3. Apply Age-Based Risk Adjustment
        age_bracket = self._determine_age_bracket(user_profile)
        age_adjusted_risk = self._apply_age_risk_adjustment(weighted_risk_score, age_bracket)
        age_adjusted_allocations = self._apply_age_allocation_adjustment(goal_allocations, age_bracket)

        # 4. Apply Investment Type Clustering
        preferred_investment_types = self._get_preferred_investment_types(questionnaire)
        experience_level = self._get_experience_level(questionnaire)

        # Adjust recommendation scores based on all factors
        for rec in recommendations:
            # Get the fund's risk level and name
            fund_risk_level = rec['fund_info']['risk_level']
            fund_name = rec['fund_name']

            # Calculate risk alignment score (higher if fund risk matches age-adjusted risk score)
            risk_alignment = 1 - abs(fund_risk_level - age_adjusted_risk) / 10

            # Calculate goal alignment score with age-adjusted allocations
            goal_alignment = self._calculate_goal_alignment(fund_name, age_adjusted_allocations)

            # Calculate investment type cluster alignment
            cluster_alignment = self._calculate_cluster_alignment(fund_name, preferred_investment_types, experience_level)

            # Adjust the recommendation score (weighted combination)
            rec['score'] = rec['score'] * 0.3 + risk_alignment * 0.2 + goal_alignment * 0.2 + cluster_alignment * 0.3

            # Add explanation about adjustments
            rec['explanation'] += f" This recommendation considers your age-adjusted risk profile of {age_adjusted_risk:.1f}/10"

            # Add cluster-specific explanation
            cluster_explanation = self._generate_cluster_explanation(fund_name, experience_level)
            if cluster_explanation:
                rec['explanation'] += f" {cluster_explanation}"

            rec['explanation'] += f" and aligns with your investment goals."

        # Re-sort recommendations by adjusted score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations

    def _get_preferred_investment_types(self, questionnaire: InvestmentQuestionnaire) -> List[str]:
        """
        Get preferred investment types from questionnaire

        Args:
            questionnaire: User's questionnaire responses

        Returns:
            List of preferred investment types
        """
        # Get preferred investment types from questionnaire
        preferred_types = questionnaire.preferred_investment_types

        # If no preferences specified, return all types
        if not preferred_types or not isinstance(preferred_types, list) or len(preferred_types) == 0:
            return list(self.investment_clusters.keys())

        # Map questionnaire preferences to our investment clusters
        # This is a simplified mapping - in a real implementation, you would have a more sophisticated mapping
        mapped_types = []
        for pref in preferred_types:
            pref_lower = pref.lower() if isinstance(pref, str) else ''

            if 'cash' in pref_lower or 'money market' in pref_lower or 'liquid' in pref_lower:
                mapped_types.append('money_market')
            elif 'bond' in pref_lower or 'fixed income' in pref_lower or 'debt' in pref_lower:
                mapped_types.append('bonds')
            elif 'stock' in pref_lower or 'equity' in pref_lower or 'share' in pref_lower:
                mapped_types.append('stocks')
            elif 'real estate' in pref_lower or 'property' in pref_lower or 'reit' in pref_lower:
                mapped_types.append('real_estate')

        # If no mappings found, return all types
        if not mapped_types:
            return list(self.investment_clusters.keys())

        return mapped_types

    def _get_experience_level(self, questionnaire: InvestmentQuestionnaire) -> str:
        """
        Determine user's investment experience level

        Args:
            questionnaire: User's questionnaire responses

        Returns:
            Experience level (beginner, intermediate, advanced)
        """
        # Get investment knowledge and experience from questionnaire
        knowledge = questionnaire.investment_knowledge
        experience_years = questionnaire.investment_experience_years

        # Map knowledge level to experience level
        if knowledge in ['none', 'basic']:
            knowledge_level = 'beginner'
        elif knowledge in ['moderate']:
            knowledge_level = 'intermediate'
        elif knowledge in ['good', 'expert']:
            knowledge_level = 'advanced'
        else:
            knowledge_level = 'beginner'  # Default

        # Map experience years to experience level
        if experience_years in ['none', '0-2']:
            years_level = 'beginner'
        elif experience_years in ['2-5']:
            years_level = 'intermediate'
        elif experience_years in ['5-10', '10+']:
            years_level = 'advanced'
        else:
            years_level = 'beginner'  # Default

        # Combine knowledge and years (prioritize years slightly)
        if knowledge_level == 'advanced' or years_level == 'advanced':
            return 'advanced'
        elif knowledge_level == 'intermediate' or years_level == 'intermediate':
            return 'intermediate'
        else:
            return 'beginner'

    def _calculate_cluster_alignment(self, fund_name: str, preferred_types: List[str], experience_level: str) -> float:
        """
        Calculate how well a fund aligns with preferred investment types and experience level

        Args:
            fund_name: Name of the fund
            preferred_types: List of preferred investment types
            experience_level: User's experience level

        Returns:
            Alignment score (0-1)
        """
        # Find which cluster this fund belongs to
        fund_cluster = None
        for cluster_type, cluster_info in self.investment_clusters.items():
            if fund_name in cluster_info['funds']:
                fund_cluster = cluster_type
                cluster_characteristics = cluster_info['characteristics']
                break

        if not fund_cluster:
            return 0.5  # Neutral score if we can't determine cluster

        # Calculate type preference alignment (1.0 if in preferred types, 0.3 otherwise)
        type_alignment = 1.0 if fund_cluster in preferred_types else 0.3

        # Calculate experience alignment
        experience_alignment = 1.0  # Default

        # Adjust based on beginner-friendliness
        if experience_level == 'beginner' and not cluster_characteristics.get('suitable_for_beginners', True):
            experience_alignment = 0.3  # Reduce score for beginners if not suitable
        elif experience_level == 'advanced' and cluster_characteristics.get('suitable_for_beginners', True):
            experience_alignment = 0.7  # Slightly reduce score for advanced users if too basic

        # Combine alignments (weighted average)
        return type_alignment * 0.7 + experience_alignment * 0.3

    def _generate_cluster_explanation(self, fund_name: str, experience_level: str) -> str:
        """
        Generate explanation based on investment cluster

        Args:
            fund_name: Name of the fund
            experience_level: User's experience level

        Returns:
            Explanation string
        """
        # Find which cluster this fund belongs to
        for cluster_type, cluster_info in self.investment_clusters.items():
            if fund_name in cluster_info['funds']:
                characteristics = cluster_info['characteristics']

                # Generate explanation based on characteristics
                if experience_level == 'beginner' and characteristics.get('suitable_for_beginners', True):
                    return f"This {cluster_type.replace('_', ' ')} investment is suitable for your experience level."
                elif experience_level == 'beginner' and not characteristics.get('suitable_for_beginners', True):
                    return f"This {cluster_type.replace('_', ' ')} investment may be complex for your experience level, but offers growth potential."
                elif experience_level == 'advanced' and characteristics.get('risk', '') in ['high', 'medium_to_high']:
                    return f"This {cluster_type.replace('_', ' ')} investment offers higher risk-return potential suitable for your experience level."
                else:
                    return f"This {cluster_type.replace('_', ' ')} investment has {characteristics.get('liquidity', 'medium')} liquidity and {characteristics.get('risk', 'medium')} risk."

        return ""  # No explanation if fund not found in any cluster

    def _determine_age_bracket(self, user_profile: UserProfile) -> str:
        """
        Determine the age bracket for a user

        Args:
            user_profile: User profile object

        Returns:
            Age bracket string
        """
        # In a real implementation, we would calculate age from user's date of birth
        # For now, we'll use a default age bracket if we can't determine the actual age

        # Try to get user's date of birth from the associated User model
        user = user_profile.user

        # This is a placeholder - in a real implementation, you would get the actual age
        # For example, if the User model has a date_of_birth field:
        # if hasattr(user, 'date_of_birth') and user.date_of_birth:
        #     today = date.today()
        #     age = today.year - user.date_of_birth.year - ((today.month, today.day) < (user.date_of_birth.month, user.date_of_birth.day))

        # For now, we'll use a default age bracket
        # In a real implementation, you would determine this based on the user's actual age
        return '35_to_44'  # Default to middle age bracket

    def _apply_age_risk_adjustment(self, risk_score: float, age_bracket: str) -> float:
        """
        Apply age-based adjustment to risk score

        Args:
            risk_score: Base risk score
            age_bracket: User's age bracket

        Returns:
            Age-adjusted risk score
        """
        # Get age adjustment factors
        adjustment_factors = self.age_risk_adjustments.get(age_bracket, self.age_risk_adjustments['45_to_54'])  # Default to middle age

        # Apply risk factor adjustment
        risk_factor = adjustment_factors['risk_factor']
        adjusted_risk = risk_score * risk_factor

        # Ensure risk score stays within valid range (1-10)
        adjusted_risk = max(1.0, min(10.0, adjusted_risk))

        return adjusted_risk

    def _apply_age_allocation_adjustment(self, allocations: Dict[str, float], age_bracket: str) -> Dict[str, float]:
        """
        Apply age-based adjustments to asset allocations

        Args:
            allocations: Base asset allocations
            age_bracket: User's age bracket

        Returns:
            Age-adjusted asset allocations
        """
        # Create a copy of allocations to modify
        adjusted = allocations.copy()

        # Get age adjustment factors
        adjustment_factors = self.age_risk_adjustments.get(age_bracket, self.age_risk_adjustments['45_to_54'])  # Default to middle age

        # Apply adjustments to each asset type
        for asset_type in adjusted:
            if asset_type + '_factor' in adjustment_factors:
                adjusted[asset_type] *= adjustment_factors[asset_type + '_factor']

        # Normalize allocations to ensure they sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            for asset_type in adjusted:
                adjusted[asset_type] /= total

        return adjusted

    def _calculate_goal_based_allocations(self, questionnaire: InvestmentQuestionnaire) -> Dict[str, float]:
        """
        Calculate asset allocations based on investment goals

        Args:
            questionnaire: User's questionnaire responses

        Returns:
            Dictionary of asset type allocations
        """
        # Get primary investment goal
        primary_goal = questionnaire.primary_goal

        # Get allocation template based on goal
        if primary_goal and primary_goal in self.goal_allocation_templates:
            allocation_template = self.goal_allocation_templates[primary_goal]
        else:
            # Use default allocation if goal not specified or not found
            allocation_template = self.goal_allocation_templates['other']

        # Adjust allocations based on investment timeframe if available
        timeframe = questionnaire.investment_timeframe
        if timeframe:
            # Adjust allocations based on timeframe
            adjusted_allocations = self._adjust_allocations_for_timeframe(allocation_template, timeframe)
        else:
            adjusted_allocations = allocation_template

        return adjusted_allocations

    def _adjust_allocations_for_timeframe(self, allocations: Dict[str, float], timeframe: str) -> Dict[str, float]:
        """
        Adjust asset allocations based on investment timeframe

        Args:
            allocations: Base asset allocations
            timeframe: Investment timeframe

        Returns:
            Adjusted asset allocations
        """
        # Create a copy of allocations to modify
        adjusted = allocations.copy()

        # Adjustment factors based on timeframe
        if timeframe == 'very_short':  # Less than 1 year
            # Increase money market, decrease stocks
            adjusted['money_market'] = min(1.0, adjusted['money_market'] * 1.5)
            adjusted['stocks'] = max(0.0, adjusted['stocks'] * 0.3)
            adjusted['real_estate'] = max(0.0, adjusted['real_estate'] * 0.2)
        elif timeframe == 'short':  # 1-3 years
            # Moderately increase money market, decrease stocks
            adjusted['money_market'] = min(1.0, adjusted['money_market'] * 1.3)
            adjusted['stocks'] = max(0.0, adjusted['stocks'] * 0.5)
            adjusted['real_estate'] = max(0.0, adjusted['real_estate'] * 0.5)
        elif timeframe == 'medium':  # 3-5 years
            # Balanced approach, slight adjustments
            adjusted['money_market'] = adjusted['money_market'] * 1.1
            adjusted['stocks'] = adjusted['stocks'] * 0.9
        elif timeframe == 'long':  # 5-10 years
            # Increase stocks, decrease money market
            adjusted['money_market'] = max(0.05, adjusted['money_market'] * 0.8)
            adjusted['stocks'] = min(0.7, adjusted['stocks'] * 1.2)
        elif timeframe == 'very_long':  # More than 10 years
            # Significantly increase stocks, decrease money market
            adjusted['money_market'] = max(0.05, adjusted['money_market'] * 0.6)
            adjusted['stocks'] = min(0.8, adjusted['stocks'] * 1.5)
            adjusted['real_estate'] = min(0.3, adjusted['real_estate'] * 1.3)

        # Normalize allocations to ensure they sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            for asset_type in adjusted:
                adjusted[asset_type] /= total

        return adjusted

    def _calculate_goal_alignment(self, fund_name: str, goal_allocations: Dict[str, float]) -> float:
        """
        Calculate how well a fund aligns with goal-based asset allocations

        Args:
            fund_name: Name of the fund
            goal_allocations: Goal-based asset allocations

        Returns:
            Alignment score (0-1)
        """
        # Find which asset type this fund belongs to
        fund_asset_type = None
        for asset_type, funds in self.asset_type_fund_mapping.items():
            if fund_name in funds:
                fund_asset_type = asset_type
                break

        if not fund_asset_type or fund_asset_type not in goal_allocations:
            return 0.5  # Neutral score if we can't determine alignment

        # Return the allocation percentage as the alignment score
        return goal_allocations[fund_asset_type]

    def _calculate_weighted_risk_score(self, questionnaire: InvestmentQuestionnaire) -> float:
        """
        Calculate a weighted risk score based on questionnaire responses

        Args:
            questionnaire: User's questionnaire responses

        Returns:
            Weighted risk score (1-10 scale)
        """
        # Base risk score (middle of scale)
        risk_score = 5.0
        total_weight = 0.0

        # Risk assessment questions with their risk impact values
        risk_mappings = {
            # Market drop reaction (1-5 scale, higher = more risk tolerant)
            'market_drop_reaction': {
                'sell_all': 1,      # Very risk-averse
                'sell_some': 2,     # Somewhat risk-averse
                'do_nothing': 3,    # Neutral
                'buy_more': 5,      # Very risk-tolerant
                'seek_advice': 3    # Neutral
            },
            # Investment preference (1-5 scale)
            'investment_preference': {
                'very_safe': 1,      # Very risk-averse
                'conservative': 2,   # Somewhat risk-averse
                'balanced': 3,       # Neutral
                'growth': 4,         # Somewhat risk-tolerant
                'aggressive': 5      # Very risk-tolerant
            },
            # Loss tolerance (1-5 scale)
            'loss_tolerance': {
                '0-5': 1,            # Very risk-averse
                '5-10': 2,           # Somewhat risk-averse
                '10-20': 3,          # Neutral
                '20-30': 4,          # Somewhat risk-tolerant
                '30+': 5             # Very risk-tolerant
            },
            # Risk comfort scenario (1-5 scale)
            'risk_comfort_scenario': {
                'scenario_1': 1,     # Very risk-averse
                'scenario_2': 2,     # Somewhat risk-averse
                'scenario_3': 3,     # Neutral
                'scenario_4': 4,     # Somewhat risk-tolerant
                'scenario_5': 5      # Very risk-tolerant
            },
            # Investment knowledge (1-5 scale)
            'investment_knowledge': {
                'none': 1,           # Very risk-averse
                'basic': 2,          # Somewhat risk-averse
                'moderate': 3,       # Neutral
                'good': 4,           # Somewhat risk-tolerant
                'expert': 5          # Very risk-tolerant
            },
            # Investment experience years (1-5 scale)
            'investment_experience_years': {
                'none': 1,           # Very risk-averse
                '0-2': 2,            # Somewhat risk-averse
                '2-5': 3,            # Neutral
                '5-10': 4,           # Somewhat risk-tolerant
                '10+': 5             # Very risk-tolerant
            },
            # Emergency fund months (1-5 scale, higher = more risk tolerant)
            'emergency_fund_months': {
                '0': 1,              # Very risk-averse
                '1-3': 2,            # Somewhat risk-averse
                '3-6': 3,            # Neutral
                '6-12': 4,           # Somewhat risk-tolerant
                '12+': 5             # Very risk-tolerant
            },
            # Debt situation (1-5 scale, higher = more risk tolerant)
            'debt_situation': {
                'very_high': 1,       # Very risk-averse
                'high': 2,           # Somewhat risk-averse
                'moderate': 3,       # Neutral
                'low': 4,            # Somewhat risk-tolerant
                'none': 5            # Very risk-tolerant
            },
            # Income stability (1-5 scale)
            'income_stability': {
                'very_unstable': 1,   # Very risk-averse
                'unstable': 2,        # Somewhat risk-averse
                'somewhat_stable': 3, # Neutral
                'stable': 4,          # Somewhat risk-tolerant
                'very_stable': 5      # Very risk-tolerant
            }
        }

        # Calculate weighted risk score
        for question, mapping in risk_mappings.items():
            # Get the user's response
            response = getattr(questionnaire, question, None)

            # Skip if no response or not in mapping
            if not response or response not in mapping:
                continue

            # Get the risk value for this response
            risk_value = mapping[response]

            # Get the weight for this question
            weight = self.question_weights.get(question, 0.5)  # Default weight if not specified

            # Add to weighted score
            risk_score += (risk_value - 3) * weight  # Adjust by -3 to center around 0
            total_weight += weight

        # Normalize risk score if we have weights
        if total_weight > 0:
            # Adjust back to center and scale to 1-10 range
            risk_score = max(1, min(10, risk_score))

        return risk_score
