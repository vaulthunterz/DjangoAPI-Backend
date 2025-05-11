from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.db.models import Sum, Count
from django.contrib.admin import SimpleListFilter
from .models import Transaction, Category, SubCategory

class SubCategoryInline(admin.TabularInline):
    model = SubCategory
    extra = 1
    fields = ('name',)
    verbose_name = "Subcategory"
    verbose_name_plural = "Subcategories"

class TransactionCategoryFilter(SimpleListFilter):
    title = 'Category'
    parameter_name = 'category'

    def lookups(self, request, model_admin):
        categories = Category.objects.all()
        return [(c.id, c.name) for c in categories]

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(category__id=self.value())
        return queryset

class TransactionUserFilter(SimpleListFilter):
    title = 'User'
    parameter_name = 'user'

    def lookups(self, request, model_admin):
        users = set([t.user for t in Transaction.objects.select_related('user').all()])
        return [(u.id, u.username) for u in users]

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(user__id=self.value())
        return queryset

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'transaction_count', 'total_amount')
    search_fields = ('name',)
    inlines = [SubCategoryInline]

    def transaction_count(self, obj):
        count = Transaction.objects.filter(category=obj).count()
        return count
    transaction_count.short_description = 'Transactions'

    def total_amount(self, obj):
        total = Transaction.objects.filter(category=obj).aggregate(Sum('amount'))['amount__sum']
        return f"${total:.2f}" if total else "$0.00"
    total_amount.short_description = 'Total Amount'

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset.prefetch_related('transactions')

@admin.register(SubCategory)
class SubCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'transaction_count', 'total_amount')
    list_filter = ('category',)
    search_fields = ('name', 'category__name')
    autocomplete_fields = ['category']

    def transaction_count(self, obj):
        count = Transaction.objects.filter(subcategory=obj).count()
        return count
    transaction_count.short_description = 'Transactions'

    def total_amount(self, obj):
        total = Transaction.objects.filter(subcategory=obj).aggregate(Sum('amount'))['amount__sum']
        return f"${total:.2f}" if total else "$0.00"
    total_amount.short_description = 'Total Amount'

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ('transaction_id', 'user_link', 'merchant_name', 'formatted_amount',
                   'category_link', 'subcategory_link', 'time_of_transaction', 'transaction_type')
    list_filter = (TransactionUserFilter, TransactionCategoryFilter, 'is_expense', 'time_of_transaction')
    search_fields = ('description', 'merchant_name', 'transaction_id', 'user__username')
    readonly_fields = ('transaction_id',)
    autocomplete_fields = ['category', 'subcategory', 'user']
    date_hierarchy = 'time_of_transaction'
    list_per_page = 50
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'transaction_id', 'merchant_name', 'amount', 'is_expense')
        }),
        ('Details', {
            'fields': ('description', 'time_of_transaction')
        }),
        ('Categorization', {
            'fields': ('category', 'subcategory')
        }),
    )

    def formatted_amount(self, obj):
        color = 'red' if obj.is_expense else 'green'
        prefix = '-' if obj.is_expense else '+'
        return format_html('<span style="color: {};">{}{:.2f}</span>', color, prefix, obj.amount)
    formatted_amount.short_description = 'Amount'

    def transaction_type(self, obj):
        if obj.is_expense:
            return format_html('<span style="color: red;">Expense</span>')
        return format_html('<span style="color: green;">Income</span>')
    transaction_type.short_description = 'Type'

    def user_link(self, obj):
        url = reverse("admin:auth_user_change", args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.user.username)
    user_link.short_description = 'User'

    def category_link(self, obj):
        if obj.category:
            url = reverse("admin:expenses_category_change", args=[obj.category.id])
            return format_html('<a href="{}">{}</a>', url, obj.category.name)
        return "-"
    category_link.short_description = 'Category'

    def subcategory_link(self, obj):
        if obj.subcategory:
            url = reverse("admin:expenses_subcategory_change", args=[obj.subcategory.id])
            return format_html('<a href="{}">{}</a>', url, obj.subcategory.name)
        return "-"
    subcategory_link.short_description = 'Subcategory'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'category', 'subcategory')

# Register custom admin site header and title
admin.site.site_header = "FinTrack Admin"
admin.site.site_title = "FinTrack Admin Portal"
admin.site.index_title = "Welcome to FinTrack Admin Portal"