from django.core.management.base import BaseCommand
from django.db import transaction
from expenses.models import Category, SubCategory

class Command(BaseCommand):
    help = 'Populates the database with predefined categories and subcategories'

    def handle(self, *args, **kwargs):
        # Define categories and their subcategories
        CATEGORIES = [
            {"name": "Shopping", "subcategories": ["Clothing & Accessories", "Electronics & Gadgets", "Home Decor & Furnishings", "Books & Stationery", "Toys & Games", "Beauty & Skincare", "Sports & Outdoor Equipment", "Jewelry & Watches", "Pet Supplies", "Hobby & Craft Supplies"]},
            {"name": "Food & Dining", "subcategories": ["Restaurants", "Fast Food", "Cafés & Coffee Shops", "Bars & Nightclubs", "Takeout & Delivery", "Groceries", "Specialty Foods", "Meal Kits & Subscriptions"]},
            {"name": "Transportation", "subcategories": ["Fuel & Gasoline", "Public Transit", "Taxi & Ride-sharing", "Airfare & Flights", "Car Maintenance & Repairs", "Parking Fees", "Road Tolls", "Vehicle Registration & Licensing", "Vehicle Leasing & Rentals", "Bike & Scooter Rentals"]},
            {"name": "Housing & Home Expenses", "subcategories": ["Rent", "Mortgage Payments", "Home Insurance", "Property Taxes", "Home Maintenance & Repairs", "Home Cleaning Services", "Pest Control", "Home Security Systems", "Landscaping & Gardening", "Moving & Relocation"]},
            {"name": "Utilities", "subcategories": ["Electricity", "Water", "Gas", "Internet Service", "Mobile Phone Bills", "Cable & Streaming Services", "Waste Management & Recycling"]},
            {"name": "Leisure & Entertainment", "subcategories": ["Movie Tickets", "Concerts & Live Events", "Theater & Performing Arts", "Amusement Parks & Arcades", "Sporting Events", "Hobbies", "Board Games & Puzzles", "Streaming Services", "Books & Magazines", "Video Games & Online Gaming"]},
            {"name": "Travel & Vacation", "subcategories": ["Flights", "Hotels & Lodging", "Car Rentals", "Travel Insurance", "Tours & Excursions", "Souvenirs & Gifts from Travel", "Visa & Passport Fees", "Travel Accessories"]},
            {"name": "Health & Fitness", "subcategories": ["Gym Memberships", "Fitness Classes", "Medical Bills", "Prescription Medications", "Over-the-Counter Medicines", "Supplements & Vitamins", "Dental Care", "Vision Care", "Physical Therapy", "Mental Health Therapy", "Alternative Medicine", "Wellness & Spa Treatments"]},
            {"name": "Education & Learning", "subcategories": ["Tuition Fees", "School Supplies", "Online Courses", "Books & Educational Materials", "Workshops & Seminars", "Student Loans Repayment", "Test Preparation & Exams Fees", "Educational Subscriptions", "Professional Certifications"]},
            {"name": "Personal Care & Grooming", "subcategories": ["Haircuts & Barber Services", "Salons & Beauty Treatments", "Skincare & Cosmetics", "Manicure & Pedicure", "Spas & Massages", "Tattoos & Piercings", "Perfumes & Fragrances", "Personal Hygiene Products"]},
            {"name": "Financial Expenses", "subcategories": ["Bank Fees", "Credit Card Annual Fees", "Loan Repayments", "Investment Contributions", "Insurance Premiums", "Taxes", "Retirement Savings"]},
            {"name": "Gifts & Donations", "subcategories": ["Personal Gifts", "Charitable Donations", "Religious Donations", "Wedding & Birthday Gifts", "Holiday Gifts", "Fundraisers & Contributions", "Tips & Gratuities"]},
            {"name": "Business & Work Expenses", "subcategories": ["Office Supplies", "Software & SaaS Subscriptions", "Business Travel Expenses", "Client Meals & Entertainment", "Advertising & Marketing", "Website Hosting & Domains", "Professional Services", "Business Insurance", "Freelancer Payments"]},
            {"name": "Childcare & Parenting", "subcategories": ["Babysitting & Nanny Services", "Daycare & Preschool Fees", "Children's Clothing & Shoes", "Kids' Toys & Games", "School Fees & Tuition", "Extracurricular Activities", "Children's Health & Medical Care"]},
            {"name": "Pets & Animal Care", "subcategories": ["Pet Food & Treats", "Veterinary Bills & Pet Insurance", "Grooming & Pet Spa Services", "Pet Boarding & Sitting", "Pet Toys & Accessories", "Pet Medications"]},
            {"name": "Legal & Professional Services", "subcategories": ["Lawyer Fees", "Notary Services", "Court Fees", "Financial Advisors", "Tax Preparation Services", "Licensing & Permits"]},
            {"name": "Miscellaneous & Unexpected Expenses", "subcategories": ["Emergency Expenses", "Repairs & Maintenance", "Subscriptions & Memberships", "Lost or Stolen Items", "Fines & Penalties"]},
            {"name": "Income & Earnings", "subcategories": ["Salary & Wages", "Freelance Income", "Investment Income", "Rental Income", "Side Hustle Earnings", "Royalties & Licensing Income", "Refunds & Rebates", "Bonuses & Commissions", "Gifts Received"]}
        ]

        # Add Unknown category and Other subcategory for fallback
        CATEGORIES.append({
            "name": "Unknown",
            "subcategories": ["Other"]
        })

        self.stdout.write('Starting to populate categories and subcategories...')

        # Counter for created items
        categories_created = 0
        subcategories_created = 0

        try:
            # First, clear existing categories and subcategories
            self.stdout.write('Clearing existing categories and subcategories...')
            SubCategory.objects.all().delete()
            Category.objects.all().delete()
            self.stdout.write('Existing data cleared.')

            # Create categories and subcategories within a transaction
            with transaction.atomic():
                for cat_data in CATEGORIES:
                    self.stdout.write(f'Processing category: {cat_data["name"]}')
                    
                    # Create category
                    category = Category.objects.create(name=cat_data['name'])
                    categories_created += 1
                    self.stdout.write(f'Created category: {category.name}')
                    
                    # Create subcategories
                    for subcat_name in cat_data['subcategories']:
                        subcategory = SubCategory.objects.create(
                            name=subcat_name,
                            category=category
                        )
                        subcategories_created += 1
                        self.stdout.write(f'Created subcategory: {subcategory.name} under {category.name}')

            self.stdout.write(self.style.SUCCESS(
                f'Successfully populated database with {categories_created} new categories '
                f'and {subcategories_created} new subcategories'
            ))

            # Verify the data
            total_categories = Category.objects.count()
            total_subcategories = SubCategory.objects.count()
            self.stdout.write(f'Verification - Total categories in database: {total_categories}')
            self.stdout.write(f'Verification - Total subcategories in database: {total_subcategories}')

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error occurred: {str(e)}'))
            raise 