from rest_framework.pagination import PageNumberPagination, LimitOffsetPagination
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


class LargeResultsSetPagination(PageNumberPagination):
    """
    Pagination class for endpoints that may return larger datasets.
    """
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 200


class TransactionPagination(PageNumberPagination):
    """
    Specialized pagination for transaction endpoints with additional metadata.
    """
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

    def get_paginated_response(self, data):
        # Calculate total amount for the current page
        total_amount = sum(float(item['amount']) for item in data if item.get('amount'))
        
        return Response({
            'count': self.page.paginator.count,
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'total_pages': self.page.paginator.num_pages,
            'current_page': self.page.number,
            'page_total_amount': total_amount,
            'results': data
        })


class CustomLimitOffsetPagination(LimitOffsetPagination):
    """
    Limit-offset based pagination for APIs that need this style.
    """
    default_limit = 20
    max_limit = 100
    limit_query_param = 'limit'
    offset_query_param = 'offset'
