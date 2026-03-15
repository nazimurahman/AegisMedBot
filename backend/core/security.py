"""
Security utilities for AegisMedBot.
Handles password hashing, JWT token creation/validation, and encryption.
Implements industry-standard security practices for healthcare data.
"""

from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
import bcrypt
import secrets
import string
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from .config import settings
import logging

logger = logging.getLogger(__name__)

# Password hashing context using bcrypt
# This provides automatic salting and hash upgrading
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Work factor for bcrypt
)

class SecurityManager:
    """
    Central security manager for all security operations.
    Handles authentication, encryption, and token management.
    """
    
    def __init__(self):
        """Initialize security manager with encryption keys."""
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        
        # Initialize encryption for PHI data
        self._init_encryption()
    
    def _init_encryption(self):
        """
        Initialize encryption for sensitive data.
        Derives encryption key from master secret using PBKDF2.
        """
        # Generate a deterministic key from the secret
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"aegismedbot_salt",  # In production, use random salt per deployment
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(settings.SECRET_KEY.encode()))
        self.cipher = Fernet(key)
    
    # Password Management
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain password against a hashed password.
        
        Args:
            plain_password: The password to verify
            hashed_password: The stored hash to compare against
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False
    
    def get_password_hash(self, password: str) -> str:
        """
        Hash a password for secure storage.
        
        Args:
            password: The password to hash
            
        Returns:
            Hashed password string
        """
        return pwd_context.hash(password)
    
    def validate_password_strength(self, password: str) -> tuple[bool, str]:
        """
        Validate password against security requirements.
        
        Args:
            password: The password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < settings.PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters long"
        
        if settings.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if settings.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        if settings.PASSWORD_REQUIRE_NUMBERS and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
        
        if settings.PASSWORD_REQUIRE_SPECIAL and not any(c in string.punctuation for c in password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is valid"
    
    # JWT Token Management
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Claims to include in the token
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        
        # Set expiration time
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),  # Issued at
            "type": "access"
        })
        
        # Add standard JWT claims
        if "sub" not in to_encode:
            to_encode["sub"] = "anonymous"
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT refresh token with longer expiration.
        
        Args:
            data: Claims to include in the token
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        
        # Set expiration time (longer than access token)
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: The JWT token to decode
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except JWTError as e:
            logger.warning(f"Token decode error: {str(e)}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """
        Create a new access token from a valid refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Dictionary with new access token and expiration, or None if invalid
        """
        payload = self.decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        # Check if refresh token is expired
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            return None
        
        # Create new access token
        new_access_token = self.create_access_token(
            data={
                "sub": payload.get("sub"),
                "role": payload.get("role"),
                "permissions": payload.get("permissions", [])
            }
        )
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    # Encryption for PHI
    def encrypt_phi(self, data: str) -> str:
        """
        Encrypt Protected Health Information (PHI).
        
        Args:
            data: String data to encrypt
            
        Returns:
            Encrypted string (base64 encoded)
        """
        if not data:
            return data
        
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"PHI encryption error: {str(e)}")
            raise ValueError("Failed to encrypt sensitive data")
    
    def decrypt_phi(self, encrypted_data: str) -> str:
        """
        Decrypt Protected Health Information (PHI).
        
        Args:
            encrypted_data: Encrypted string to decrypt
            
        Returns:
            Decrypted string
        """
        if not encrypted_data:
            return encrypted_data
        
        try:
            decrypted = self.cipher.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"PHI decryption error: {str(e)}")
            raise ValueError("Failed to decrypt sensitive data")
    
    # API Key Management
    def generate_api_key(self) -> str:
        """
        Generate a secure random API key.
        
        Returns:
            API key string
        """
        # Generate 32 random bytes and encode as base64
        key_bytes = secrets.token_bytes(32)
        api_key = base64.urlsafe_b64encode(key_bytes).decode().rstrip("=")
        
        # Prefix with aead for identification
        return f"aeg_{api_key}"
    
    def hash_api_key(self, api_key: str) -> str:
        """
        Hash an API key for storage.
        
        Args:
            api_key: The API key to hash
            
        Returns:
            Hashed API key
        """
        # Use SHA256 for API key hashing (not for passwords)
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    # Secure Token Generation
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token.
        
        Args:
            length: Length of the token in bytes
            
        Returns:
            Hex-encoded random token
        """
        return secrets.token_hex(length)
    
    def generate_temporary_password(self) -> str:
        """
        Generate a secure temporary password meeting requirements.
        
        Returns:
            Random password string
        """
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        while True:
            password = "".join(secrets.choice(alphabet) for _ in range(settings.PASSWORD_MIN_LENGTH))
            # Validate against requirements
            is_valid, _ = self.validate_password_strength(password)
            if is_valid:
                return password
    
    # CSRF Protection
    def generate_csrf_token(self) -> str:
        """
        Generate a CSRF token for form protection.
        
        Returns:
            CSRF token string
        """
        return secrets.token_urlsafe(32)
    
    def verify_csrf_token(self, token: str, stored_token: str) -> bool:
        """
        Verify a CSRF token using constant-time comparison.
        
        Args:
            token: Token to verify
            stored_token: Stored token to compare against
            
        Returns:
            True if tokens match
        """
        return hmac.compare_digest(token, stored_token)
    
    # Rate Limiting Helpers
    def get_rate_limit_key(self, identifier: str, action: str) -> str:
        """
        Generate a Redis key for rate limiting.
        
        Args:
            identifier: User or IP identifier
            action: The action being rate limited
            
        Returns:
            Redis key string
        """
        return f"rate_limit:{action}:{identifier}"
    
    # Audit Trail Helpers
    def mask_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """
        Mask sensitive data in logs.
        
        Args:
            data: Dictionary containing data to mask
            sensitive_fields: List of field names to mask
            
        Returns:
            Dictionary with sensitive fields masked
        """
        masked_data = data.copy()
        for field in sensitive_fields:
            if field in masked_data and masked_data[field]:
                # Mask all but last 4 characters
                value = str(masked_data[field])
                if len(value) > 4:
                    masked_data[field] = "*" * (len(value) - 4) + value[-4:]
                else:
                    masked_data[field] = "*" * len(value)
        return masked_data

# Create global security manager instance
security_manager = SecurityManager()

# Convenience functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Convenience function for password verification."""
    return security_manager.verify_password(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Convenience function for password hashing."""
    return security_manager.get_password_hash(password)

def create_access_token(data: Dict[str, Any]) -> str:
    """Convenience function for creating access tokens."""
    return security_manager.create_access_token(data)

def encrypt_phi(data: str) -> str:
    """Convenience function for PHI encryption."""
    return security_manager.encrypt_phi(data)

def decrypt_phi(data: str) -> str:
    """Convenience function for PHI decryption."""
    return security_manager.decrypt_phi(data)