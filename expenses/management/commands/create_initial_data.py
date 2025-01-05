from django.core.management.base import BaseCommand
from expenses.models import Category, SubCategory

class Command(BaseCommand):
    help = 'Creates initial categories and subcategories.'

    def handle(self, *args, **kwargs):
        categories_data = [
            { "name": "Shopping", "subcategories": [ "Clothing", "Electronics", "Home Goods", "Furniture", "Books", "Toys", "Cosmetics", "Sports Equipment", "Pharmacy" ] },
            { "name": "Food & Dining", "subcategories": [ "Restaurants", "Fast Food", "Coffee Shops", "Bars & Pubs", "Cafes", "Takeout Delivery", "Groceries" ] },
            { "name": "Transportation", "subcategories": [ "Fuel", "Public Transit", "Taxi", "Air Travel", "Car Maintenance", "Parking", "Tolls", "Bike Rentals" ] },
            { "name": "Housing", "subcategories": [ "Rent","Mortgage", "Internet", "Home Maintenance", "Home Insurance", "Furniture","Appliances", "Cleaning Services" ] },
            { "name": "Leisure & Entertainment", "subcategories": [ "Movies"," Theaters", "Concert Events", "Streaming Services", "Hobbies", "Gaming", "Sports Activities", "Exhibitions", "Amusement Parks" ] },
            { "name": "Travel", "subcategories": [ "Flights", "Hotels", "Car Rentals", "Travel Insurance", "Souvenirs", "Tours", "Passport Fees" ] },
            { "name": "Health & Fitness", "subcategories": [ "Gym Membership", "Fitness Classes", "Medical Expenses", "Health Insurance", "Pharmacy", "Supplements", "Wellness" ] },
            { "name": "Education", "subcategories": [ "Tuition Fees", "Books","School Supplies", "Online Courses", "Workshops", "Student Loans", "School Fees" ] },
            { "name": "Personal Care", "subcategories": [ "Haircuts","Salons", "Skincare", "Cosmetics", "Spa", "Tattoos","Piercings" ] },
            { "name": "Utilities", "subcategories": [ "Electricity", "Water", "Gas", "Internet", "Mobile Phone", "Cable TV", "Waste Collection" ] },
            { "name": "Financial", "subcategories": [ "Bank Fees", "Credit Card Fees", "Investments", "Loans", "Insurance", "Taxes" ] },
            { "name": "Gifts & Donations", "subcategories": [ "Gifts", "Charitable Donations", "Tips" ] },
            { "name": "Business Expenses", "subcategories": [ "Office Supplies", "Software", "Travel", "Client Meals", "Advertising",  "Professional Services" ] },
            { "name": "Miscellaneous", "subcategories": [ "Uncategorized", "Pet Expenses", "Legal Fees", "Subscriptions", "Repairs" ] },
            { "name": "Income", "subcategories": [ "Salary", "Freelance Income", "Investment Income", "Gifts Received", "Refunds" ] }
        ]
        for category_data in categories_data:
            category_name = category_data['name']
            subcategories = category_data['subcategories']

            category, created = Category.objects.get_or_create(name=category_name)
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created category: {category_name}'))

            for subcategory_name in subcategories:
                subcategory, created = SubCategory.objects.get_or_create(name=subcategory_name, category=category)
                if created:
                    self.stdout.write(
                        self.style.SUCCESS(f'Created subcategory: {subcategory_name} under {category_name}'))
        self.stdout.write(self.style.SUCCESS('All default categories and subcategories have been created!'))

