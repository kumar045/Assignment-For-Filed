from rest_framework import serializers


class PaymentGatewaySerializer(serializers.Serializer):
    CreditCardNumber = serializers.CharField()
    CardHolder = serializers.CharField()
    ExpirationDate = serializers.DateField()
    SecurityCode = serializers.CharField()
    Amount = serializers.FloatField()
