from django.db import models
from django.contrib.auth.models import User  # Import User model

class Category(models.Model):
    name = models.CharField(max_length=200, unique=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Categories"

class SubCategory(models.Model):
    name = models.CharField(max_length=200)
    category = models.ForeignKey(Category, on_delete=models.CASCADE,related_name='subcategories')

    class Meta:
        unique_together = (
        'name', 'category')  # enforce uniqueness of subcategories in the context of the parent category
        verbose_name_plural = "Subcategories"

    def __str__(self):
        return f"{self.category.name} - {self.name}"


class Transaction(models.Model):
    # Link each transaction to a specific user
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='transactions')
    transaction_id = models.CharField(max_length=100, unique=True)
    merchant_name = models.CharField(max_length=255, blank=True, null=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField(default='')  # Add default empty string
    time_of_transaction = models.DateTimeField()
    is_expense = models.BooleanField(default=True)  # True for expense, False for income
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, blank=True, null=True, related_name='transactions')
    subcategory = models.ForeignKey(SubCategory, on_delete=models.SET_NULL, blank=True, null=True, related_name='transactions')

    def __str__(self):
        return f"{self.user.username} - {self.merchant_name} - {self.amount}" #Show username

    class Meta:
        ordering = [ '-time_of_transaction' ]  # Show newest transactions first