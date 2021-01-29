from django.conf.urls import url
from .views import *
from django.urls import path

urlpatterns = [
    url(r'^payment_process/$', PaymentGatewayAPIView.as_view(), name='payment_process'),


]