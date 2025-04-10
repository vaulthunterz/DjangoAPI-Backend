"""
Hybrid recommender system that combines rule-based and ML approaches
"""
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from django.db.models import Sum, Count
from django.utils import timezone
from datetime import timedelta

from ...models import UserProfile, MoneyMarketFund as FinancialAsset, Portfolio, PortfolioItem, Recommendation, InvestmentQuestionnaire
from expenses.models import Transaction, Category, SubCategory
from .base import BaseRecommender

class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender that combines rule-based and ML approaches,
    and leverages expense categorization data for personalized recommendations.
    """

    def __init__(self):
        self.risk_level_mapping = {
            'low': 1,
            'medium': 5,
            'high': 10
        }

        # Kenyan Money Market Funds with their risk levels (1-10 scale)
        self.money_market_funds = {
            'CIC Money Market Fund': {
                'risk_level': 2,
                'description': 'A low-risk fund managed by CIC Asset Management',
                'min_investment': 1000,
                'returns': '7-9% annually'
            },
            'NCBA Money Market Fund': {
                'risk_level': 2,
                'description': 'A stable fund managed by NCBA Investment Bank',
                'min_investment': 1000,
                'returns': '7-9% annually'
            },
            'Jubilee Money Market Fund': {
                'risk_level': 2,
                'description': 'A conservative fund managed by Jubilee Insurance',
                'min_investment': 1000,
                'returns': '7-9% annually'
            },
            'Sanlam Money Market Fund': {
                'risk_level': 2,
                'description': 'A low-risk fund managed by Sanlam Investments',
                'min_investment': 1000,
                'returns': '7-9% annually'
            },
            'Old Mutual Money Market Fund': {
                'risk_level': 2,
                'description': 'A stable fund managed by Old Mutual Investment Group',
                'min_investment': 1000,
                'returns': '7-9% annually'
            },
            'Britam Money Market Fund': {
                'risk_level': 2,
                'description': 'A conservative fund managed by Britam Asset Managers',
                'min_investment': 1000,
                'returns': '7-9% annually'
            },
            'Cytonn Money Market Fund': {
                'risk_level': 2,
                'description': 'A low-risk fund managed by Cytonn Investments',
                'min_investment': 1000,
                'returns': '7-9% annually'
            },
            'Genghis Capital Money Market Fund': {
                'risk_level': 2,
                'description': 'A stable fund managed by Genghis Capital',
                'min_investment': 1000,
                'returns': '7-9% annually'
            },
            'Nabo Capital Money Market Fund': {
                'risk_level': 2,
                'description': 'A conservative fund managed by Nabo Capital',
                'min_investment': 1000,
                'returns': '7-9% annually'
            },
            'Zimele Money Market Fund': {
                'risk_level': 2,
                'description': 'A low-risk fund managed by Zimele Asset Management',
                'min_investment': 1000,
                'returns': '7-9% annually'
            }
        }

        # Investment timeline to fund selection mapping
        self.timeline_fund_mapping = {
            'short': ['CIC Money Market Fund', 'NCBA Money Market Fund', 'Jubilee Money Market Fund'],
            'mid': ['Sanlam Money Market Fund', 'Old Mutual Money Market Fund', 'Britam Money Market Fund'],
            'long': ['Cytonn Money Market Fund', 'Genghis Capital Money Market Fund', 'Nabo Capital Money Market Fund', 'Zimele Money Market Fund']
        }

        # Experience level to fund selection mapping
        self.experience_fund_mapping = {
            'beginner': ['CIC Money Market Fund', 'NCBA Money Market Fund', 'Jubilee Money Market Fund'],
            'intermediate': ['Sanlam Money Market Fund', 'Old Mutual Money Market Fund', 'Britam Money Market Fund'],
            'advanced': ['Cytonn Money Market Fund', 'Genghis Capital Money Market Fund', 'Nabo Capital Money Market Fund', 'Zimele Money Market Fund']
        }

    def train(self, training_data: Any) -> None:
        """Not needed for rule-based component, but required by interface"""
        pass

    def predict(self, user_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations based on user features

        Args:
            user_features: Dictionary containing user profile information

        Returns:
            List of recommendation dictionaries
        """
        # Extract user profile information
        user_profile = user_features.get('user_profile')
        questionnaire = user_features.get('questionnaire')

        if not user_profile:
            return []

        # Get rule-based recommendations
        rule_based_recs = self._get_rule_based_recommendations(user_profile, questionnaire)

        # Get expense-based recommendations
        expense_based_recs = self._get_expense_based_recommendations(user_profile)

        # Combine recommendations with weights
        combined_recs = self._combine_recommendations(rule_based_recs, expense_based_recs)

        return combined_recs

    def _get_rule_based_recommendations(self, user_profile: UserProfile, questionnaire: InvestmentQuestionnaire = None) -> List[Dict[str, Any]]:
        """Generate rule-based recommendations based on user profile"""
        recommendations = []

        # Get user's risk tolerance
        risk_tolerance = user_profile.risk_tolerance

        # Get user's investment timeline
        investment_timeline = user_profile.investment_timeline

        # Get user's investment experience
        investment_experience = user_profile.investment_experience

        # Get questionnaire data if available
        risk_score = None
        if questionnaire:
            risk_score = questionnaire.risk_tolerance_score
            # Use questionnaire risk score if available
            if risk_score:
                risk_tolerance = min(3, max(1, (risk_score // 4) + 1))

        # Get suitable funds based on timeline and experience
        suitable_funds = list(set(
            self.timeline_fund_mapping.get(investment_timeline, list(self.money_market_funds.keys())) +
            self.experience_fund_mapping.get(investment_experience, list(self.money_market_funds.keys()))
        ))

        # Generate recommendations for each suitable fund
        for fund_name in suitable_funds:
            fund_info = self.money_market_funds[fund_name]

            # Skip if fund risk level too high for user
            if fund_info['risk_level'] > risk_tolerance * 3:  # Scale risk tolerance (1-3) to match fund risk (1-10)
                continue

            # Calculate recommendation score
            score = self._calculate_rule_based_score(fund_name, fund_info, risk_tolerance, investment_timeline, investment_experience)

            # Generate explanation
            explanation = self._generate_rule_based_explanation(fund_name, fund_info, risk_tolerance, investment_timeline, investment_experience)

            recommendations.append({
                'fund_name': fund_name,
                'fund_info': fund_info,
                'score': score,
                'explanation': explanation,
                'recommendation_type': 'rule_based'
            })

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations

    def _get_expense_based_recommendations(self, user_profile: UserProfile) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on user's expense patterns

        This leverages the expense categorization data to understand user's spending habits
        and preferences, which can inform investment recommendations.
        """
        recommendations = []

        # Get user's transactions
        user = user_profile.user
        transactions = Transaction.objects.filter(user=user)

        if not transactions.exists():
            return []

        # Analyze expense categories
        category_distribution = self._analyze_expense_categories(transactions)

        # Analyze income sources
        income_sources = self._analyze_income_sources(transactions)

        # Generate recommendations for each money market fund
        for fund_name, fund_info in self.money_market_funds.items():
            # Calculate recommendation score based on expense patterns
            score = self._calculate_expense_based_score(fund_name, fund_info, category_distribution, income_sources)

            # Skip if score too low
            if score < 0.3:
                continue

            # Generate explanation
            explanation = self._generate_expense_based_explanation(fund_name, fund_info, category_distribution, income_sources)

            recommendations.append({
                'fund_name': fund_name,
                'fund_info': fund_info,
                'score': score,
                'explanation': explanation,
                'recommendation_type': 'expense_based'
            })

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations

    def _analyze_expense_categories(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Analyze user's expense categories to understand spending patterns"""
        # Filter for expenses only
        expenses = transactions.filter(is_expense=True)

        if not expenses.exists():
            return {}

        # Group by category and calculate total amount
        category_totals = {}
        total_amount = 0

        for expense in expenses:
            if expense.category:
                category_name = expense.category.name
                amount = float(expense.amount)

                if category_name in category_totals:
                    category_totals[category_name] += amount
                else:
                    category_totals[category_name] = amount

                total_amount += amount

        # Calculate percentages
        category_distribution = {}
        if total_amount > 0:
            for category, amount in category_totals.items():
                category_distribution[category] = amount / total_amount

        return category_distribution

    def _analyze_income_sources(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Analyze user's income sources"""
        # Filter for income only
        incomes = transactions.filter(is_expense=False)

        if not incomes.exists():
            return {}

        # Group by category and calculate total amount
        source_totals = {}
        total_amount = 0

        for income in incomes:
            if income.category:
                category_name = income.category.name
                amount = float(income.amount)

                if category_name in source_totals:
                    source_totals[category_name] += amount
                else:
                    source_totals[category_name] = amount

                total_amount += amount

        # Calculate percentages
        source_distribution = {}
        if total_amount > 0:
            for source, amount in source_totals.items():
                source_distribution[source] = amount / total_amount

        return source_distribution

    def _calculate_rule_based_score(self, fund_name: str, fund_info: Dict[str, Any], risk_tolerance: int,
                                   investment_timeline: str, investment_experience: str) -> float:
        """Calculate recommendation score based on rules"""
        score = 0.5  # Base score

        # Risk alignment (higher score if fund risk matches user risk tolerance)
        risk_alignment = 1 - abs(fund_info['risk_level'] - (risk_tolerance * 3)) / 10
        score += risk_alignment * 0.3

        # Timeline alignment
        if fund_name in self.timeline_fund_mapping.get(investment_timeline, []):
            score += 0.2

        # Experience alignment
        if fund_name in self.experience_fund_mapping.get(investment_experience, []):
            score += 0.2

        return min(1.0, score)

    def _calculate_expense_based_score(self, fund_name: str, fund_info: Dict[str, Any],
                                      category_distribution: Dict[str, float],
                                      income_sources: Dict[str, float]) -> float:
        """Calculate recommendation score based on expense patterns"""
        score = 0.3  # Base score

        # Map fund managers to expense categories (simplified mapping)
        manager_category_mapping = {
            'CIC': ['Insurance', 'Healthcare', 'Education'],
            'NCBA': ['Banking', 'Finance', 'Business'],
            'Jubilee': ['Insurance', 'Healthcare', 'Family'],
            'Sanlam': ['Insurance', 'Healthcare', 'Retirement'],
            'Old Mutual': ['Insurance', 'Retirement', 'Family'],
            'Britam': ['Insurance', 'Healthcare', 'Family'],
            'Cytonn': ['Real Estate', 'Development', 'Business'],
            'Genghis': ['Banking', 'Finance', 'Business'],
            'Nabo': ['Banking', 'Finance', 'Business'],
            'Zimele': ['Development', 'Business', 'Education']
        }

        # Find matching categories for this fund's manager
        matching_categories = []
        for manager, categories in manager_category_mapping.items():
            if manager.lower() in fund_name.lower():
                matching_categories.extend(categories)

        # If no matching categories, return base score
        if not matching_categories:
            return score

        # Calculate score based on matching categories
        matching_score = 0
        for category, percentage in category_distribution.items():
            if any(match.lower() in category.lower() for match in matching_categories):
                matching_score += percentage

        # Add matching score to base score
        score += matching_score * 0.7

        return min(1.0, score)

    def _generate_rule_based_explanation(self, fund_name: str, fund_info: Dict[str, Any], risk_tolerance: int,
                                        investment_timeline: str, investment_experience: str) -> str:
        """Generate explanation for rule-based recommendation"""
        reasons = []

        # Risk alignment
        if fund_info['risk_level'] <= risk_tolerance * 3:
            reasons.append(f"This money market fund aligns with your risk tolerance level")

        # Timeline alignment
        if fund_name in self.timeline_fund_mapping.get(investment_timeline, []):
            reasons.append(f"Suitable for your {investment_timeline} investment horizon")

        # Experience alignment
        if fund_name in self.experience_fund_mapping.get(investment_experience, []):
            reasons.append(f"Appropriate for your {investment_experience} investment experience")

        # Add fund details
        reasons.append(f"Expected returns: {fund_info['returns']}")
        reasons.append(f"Minimum investment: KES {fund_info['min_investment']}")

        return " and ".join(reasons) if reasons else "Based on your investment profile"

    def _generate_expense_based_explanation(self, fund_name: str, fund_info: Dict[str, Any],
                                           category_distribution: Dict[str, float],
                                           income_sources: Dict[str, float]) -> str:
        """Generate explanation for expense-based recommendation"""
        # Find top matching category
        top_category = None
        top_percentage = 0

        for category, percentage in category_distribution.items():
            if any(manager.lower() in category.lower() for manager in ['CIC', 'NCBA', 'Jubilee', 'Sanlam', 'Old Mutual', 'Britam', 'Cytonn', 'Genghis', 'Nabo', 'Zimele']):
                if percentage > top_percentage:
                    top_category = category
                    top_percentage = percentage

        if top_category:
            return f"Based on your spending in {top_category} ({top_percentage:.1%} of your expenses) and the fund's expected returns of {fund_info['returns']}"

        return f"Based on your overall spending patterns and the fund's expected returns of {fund_info['returns']}"

    def _combine_recommendations(self, rule_based_recs: List[Dict[str, Any]],
                                expense_based_recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine rule-based and expense-based recommendations"""
        # Create a dictionary to store combined recommendations
        combined_recs = {}

        # Process rule-based recommendations
        for rec in rule_based_recs:
            fund_name = rec['fund_name']
            combined_recs[fund_name] = {
                'fund_name': fund_name,
                'fund_info': rec['fund_info'],
                'rule_score': rec['score'],
                'expense_score': 0,
                'combined_score': rec['score'] * 0.7,  # Weight rule-based score
                'rule_explanation': rec['explanation'],
                'expense_explanation': None,
                'recommendation_type': 'hybrid'
            }

        # Process expense-based recommendations
        for rec in expense_based_recs:
            fund_name = rec['fund_name']

            if fund_name in combined_recs:
                # Update existing recommendation
                combined_recs[fund_name]['expense_score'] = rec['score']
                combined_recs[fund_name]['expense_explanation'] = rec['explanation']
                combined_recs[fund_name]['combined_score'] = (
                    combined_recs[fund_name]['rule_score'] * 0.7 +
                    rec['score'] * 0.3
                )
            else:
                # Add new recommendation
                combined_recs[fund_name] = {
                    'fund_name': fund_name,
                    'fund_info': rec['fund_info'],
                    'rule_score': 0,
                    'expense_score': rec['score'],
                    'combined_score': rec['score'] * 0.3,  # Weight expense-based score
                    'rule_explanation': None,
                    'expense_explanation': rec['explanation'],
                    'recommendation_type': 'expense_based'
                }

        # Convert to list and sort by combined score
        result = []
        for fund_name, rec in combined_recs.items():
            # Generate combined explanation
            explanations = []
            if rec['rule_explanation']:
                explanations.append(rec['rule_explanation'])
            if rec['expense_explanation']:
                explanations.append(rec['expense_explanation'])

            combined_explanation = " ".join(explanations)

            result.append({
                'fund_name': fund_name,
                'fund_info': rec['fund_info'],
                'score': rec['combined_score'],
                'explanation': combined_explanation,
                'recommendation_type': rec['recommendation_type']
            })

        # Sort by combined score
        result.sort(key=lambda x: x['score'], reverse=True)

        return result

    def update(self, feedback_data: Any) -> None:
        """Not needed for rule-based component, but required by interface"""
        pass

    def save_model(self, path: str) -> None:
        """Not needed for rule-based component, but required by interface"""
        pass

    def load_model(self, path: str) -> None:
        """Not needed for rule-based component, but required by interface"""
        pass