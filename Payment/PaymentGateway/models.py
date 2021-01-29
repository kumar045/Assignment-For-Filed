from django.db import models


# Create your models here.
#by default charfield can't be blank so it is mandatory
class PaymentGateway(models.Model):
    CreditCardNumber=models.CharField(max_length=1500)
    CardHolder=models.CharField(max_length=1500)
    ExpirationDate=models.DateField()
    SecurityCode=models.CharField(max_length=3)
    Amount=models.FloatField()