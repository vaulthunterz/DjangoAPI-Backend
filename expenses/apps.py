from django.apps import AppConfig



class ExpensesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'expenses' # <- Verify that this equals to expenses