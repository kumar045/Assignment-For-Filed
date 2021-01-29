from rest_framework.response import Response
from rest_framework.generics import CreateAPIView
from rest_framework import status
from .models import *
from .serializers import PaymentGatewaySerializer
from datetime import datetime, date
import re


class PaymentGatewayAPIView(CreateAPIView):
    serializer_class = PaymentGatewaySerializer
    queryset = PaymentGateway.objects.all()

    def create(self, request, format=None):
        """
                Takes the request from the post and then processes the algorithm to extract the data and return the result in a
                JSON format
                :param request:
                :param format:
                :return:
                """

        serializer = self.serializer_class(data=request.data)

        if serializer.is_valid():
            CreditCardNumber = self.request.data['CreditCardNumber']
            CardHolder = self.request.data['CardHolder']
            ExpirationDate = self.request.data['ExpirationDate']
            SecurityCode= self.request.data['SecurityCode']
            Amount= self.request.data['Amount']
            content = []
            print(ExpirationDate)
            ExpirationDate = datetime.strptime(ExpirationDate, "%Y-%m-%d").date()
            now = date.today()


            pattern = '^[973][0-9]{15}|[973][0-9]{3}-[0-9]{4}-[0-9]{4}-[0-9]{4}$'
            result = re.match(pattern, CreditCardNumber)
            Payment="UnSuccess"
            if(Amount<20 and ExpirationDate >= now and result):
                Payment="Done by CheapPaymentGateway"
            elif(Amount>21 and Amount<500 and ExpirationDate >= now and result):
                Payment = "Done by ExpensivePaymentGateway"
            elif (Amount > 500 and ExpirationDate >= now and result):
                Payment = "Done by PremiumPaymentGateway"




            # add result to the dictionary and revert as response
            mydict = {
                'status': True,
                'response':
                    {

                        'Payment': Payment,

                    }
            }
            content.append(mydict)

            return Response(content, status=status.HTTP_200_OK)
        errors = serializer.errors

        response_text = {
            "status": False,
            "response": errors
        }
        return Response(response_text, status=status.HTTP_400_BAD_REQUEST)

