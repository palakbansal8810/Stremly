import bcrypt
import pyotp
from twilio.rest import Client

class AuthenticationService:
    def __init__(self):
        self.twilio_client = Client(account_sid, auth_token)
        
    def verify_password(self, username, password):
        stored_hash = self.get_user_password_hash(username)
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
    
    def send_sms_code(self, phone_number):
        code = self.generate_verification_code()
        message = self.twilio_client.messages.create(
            body=f"Your verification code is: {code}",
            from_='+1234567890',
            to=phone_number
        )
        return code
    
    def verify_totp(self, secret, token):
        totp = pyotp.TOTP(secret)
        return totp.verify(token)
    
    def login(self, username, password, mfa_code):
        if not self.verify_password(username, password):
            return False
        
        user = self.get_user(username)
        if user.mfa_enabled:
            if user.mfa_method == 'sms':
                return self.verify_sms_code(user.phone, mfa_code)
            elif user.mfa_method == 'totp':
                return self.verify_totp(user.totp_secret, mfa_code)
        
        return True
