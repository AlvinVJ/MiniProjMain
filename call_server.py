from twilio.rest import Client
import os

account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')

client = Client(account_sid, auth_token)

call = client.calls.create(
                        url='http://35.225.45.126:5000/voice',
                        to='+918921357368',
                        from_='+12568073757'
                    )

print(call.sid)

# To get the call status
call = client.calls(call.sid).fetch()
print(call.status)