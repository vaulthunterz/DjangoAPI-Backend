from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response


class StandardResultsSetPagination(PageNumberPagination):
    """
    Standard pagination class with page size of 20 and customizable page size parameter.
    """
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

    def get_paginated_response(self, data):
        return Response({
            'count': self.page.paginator.count,
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'total_pages': self.page.paginator.num_pages,
            'current_page': self.page.number,
            'results': data
        })


class PortfolioPagination(PageNumberPagination):
    """
    Specialized pagination for portfolio endpoints with additional metadata.
    """
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 50

    def get_paginated_response(self, data):
        # Calculate total value for the current page
        total_value = sum(float(item.get('total_value', 0)) for item in data if item.get('total_value'))
        
        return Response({
            'count': self.page.paginator.count,
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'total_pages': self.page.paginator.num_pages,
            'current_page': self.page.number,
            'page_total_value': total_value,
            'results': data
        })


class RecommendationPagination(PageNumberPagination):
    """
    Specialized pagination for recommendation endpoints.
    """
    page_size = 15
    page_size_query_param = 'page_size'
    max_page_size = 50
