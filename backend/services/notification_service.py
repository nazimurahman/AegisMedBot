"""
Notification service for AegisMedBot.

This module handles all notification delivery across multiple channels
including email, SMS, in-app notifications, and webhooks. It provides
a unified interface for sending notifications with templating support.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import asyncio
import logging
import json
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from redis.asyncio import Redis

from ..core.config import settings

logger = logging.getLogger(__name__)


class NotificationPriority:
    """Constants for notification priority levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel:
    """Constants for notification delivery channels."""
    
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"


class NotificationService:
    """
    Centralized notification service supporting multiple channels.
    
    Features:
    - Multi-channel delivery (email, SMS, in-app, webhooks)
    - Template-based message formatting
    - Priority-based queuing
    - Retry logic for failed deliveries
    - Delivery tracking and analytics
    """
    
    def __init__(self, redis_client: Optional[Redis] = None):
        """
        Initialize the notification service.
        
        Args:
            redis_client: Optional Redis client for queueing and caching
        """
        self.redis_client = redis_client
        
        # Email configuration
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.FROM_EMAIL
        
        # Webhook configuration
        self.webhook_timeout = 10  # seconds
        
        # Templates cache
        self.templates = {}
        self._load_templates()
        
        logger.info("NotificationService initialized")
    
    def _load_templates(self):
        """Load notification templates from configuration."""
        # In production, load from database or files
        self.templates = {
            "chat_response": {
                "subject": "New response from MedIntel Assistant",
                "email": """
                    <h2>MedIntel Assistant Response</h2>
                    <p>Your query has been processed:</p>
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 5px;">
                        <strong>Query:</strong> {{query}}<br><br>
                        <strong>Response:</strong> {{response}}
                    </div>
                    <p><a href="{{conversation_url}}">View full conversation</a></p>
                """,
                "sms": "MedIntel: {{response_preview}}"
            },
            "human_escalation": {
                "subject": "Human Intervention Required",
                "email": """
                    <h2>Human Intervention Required</h2>
                    <p>A query requires human attention:</p>
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 5px;">
                        <strong>User:</strong> {{user_name}}<br>
                        <strong>Query:</strong> {{query}}<br>
                        <strong>Suggested Agent:</strong> {{suggested_agent}}<br>
                        <strong>Confidence:</strong> {{confidence}}%
                    </div>
                    <p><a href="{{escalation_url}}">Review and respond</a></p>
                """,
                "sms": "Human intervention needed for query: {{query_preview}}"
            },
            "alert": {
                "subject": "MedIntel Alert: {{alert_type}}",
                "email": """
                    <h2>System Alert</h2>
                    <p><strong>Type:</strong> {{alert_type}}</p>
                    <p><strong>Severity:</strong> {{severity}}</p>
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 5px;">
                        {{message}}
                    </div>
                    <p><a href="{{alert_url}}">View details</a></p>
                """
            },
            "report_ready": {
                "subject": "Your MedIntel Report is Ready",
                "email": """
                    <h2>Report Ready</h2>
                    <p>Your requested report is now available:</p>
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 5px;">
                        <strong>Report Type:</strong> {{report_type}}<br>
                        <strong>Period:</strong> {{period}}<br>
                        <strong>Generated:</strong> {{generated_at}}
                    </div>
                    <p><a href="{{download_url}}">Download Report</a></p>
                """
            }
        }
    
    async def send_notification(
        self,
        user_id: str,
        template_name: str,
        template_data: Dict[str, Any],
        channels: List[str] = [NotificationChannel.IN_APP],
        priority: str = NotificationPriority.MEDIUM,
        user_contact: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send a notification to a user.
        
        Args:
            user_id: ID of the recipient user
            template_name: Name of the template to use
            template_data: Data for template rendering
            channels: List of channels to use
            priority: Notification priority
            user_contact: User's contact information
            
        Returns:
            Dictionary with delivery status per channel
        """
        result = {
            "notification_id": str(uuid4()),
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "channels": {},
            "status": "pending"
        }
        
        # Validate template
        if template_name not in self.templates:
            logger.error(f"Template not found: {template_name}")
            result["status"] = "failed"
            result["error"] = f"Template not found: {template_name}"
            return result
        
        template = self.templates[template_name]
        
        # Prepare tasks for each channel
        tasks = []
        for channel in channels:
            if channel == NotificationChannel.EMAIL:
                if user_contact and user_contact.get("email"):
                    tasks.append(self._send_email(
                        user_contact["email"],
                        self._render_template(template["subject"], template_data),
                        self._render_template(template["email"], template_data)
                    ))
                else:
                    logger.warning(f"No email for user {user_id}")
            
            elif channel == NotificationChannel.SMS:
                if user_contact and user_contact.get("phone"):
                    sms_text = self._render_template(
                        template.get("sms", template["subject"]),
                        template_data
                    )
                    tasks.append(self._send_sms(
                        user_contact["phone"],
                        sms_text
                    ))
                else:
                    logger.warning(f"No phone for user {user_id}")
            
            elif channel == NotificationChannel.IN_APP:
                tasks.append(self._send_in_app(
                    user_id,
                    template_name,
                    template_data
                ))
            
            elif channel == NotificationChannel.WEBHOOK:
                if user_contact and user_contact.get("webhook_url"):
                    tasks.append(self._send_webhook(
                        user_contact["webhook_url"],
                        template_data
                    ))
        
        # Execute all channel deliveries concurrently
        if tasks:
            channel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, channel_result in enumerate(channel_results):
                channel_name = channels[i]
                if isinstance(channel_result, Exception):
                    result["channels"][channel_name] = {
                        "status": "failed",
                        "error": str(channel_result)
                    }
                    logger.error(f"Failed to send {channel_name}: {str(channel_result)}")
                else:
                    result["channels"][channel_name] = {
                        "status": "sent",
                        "details": channel_result
                    }
        
        # Determine overall status
        if all(c.get("status") == "sent" for c in result["channels"].values()):
            result["status"] = "delivered"
        elif any(c.get("status") == "sent" for c in result["channels"].values()):
            result["status"] = "partial"
        else:
            result["status"] = "failed"
        
        # Store notification record
        if self.redis_client:
            await self.redis_client.setex(
                f"notification:{result['notification_id']}",
                604800,  # 7 days
                json.dumps(result)
            )
        
        logger.info(f"Notification {result['notification_id']} sent to user {user_id}")
        return result
    
    def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """
        Simple template rendering with variable substitution.
        
        Args:
            template: Template string with {{variable}} placeholders
            data: Dictionary of variable values
            
        Returns:
            Rendered template
        """
        result = template
        for key, value in data.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result
    
    async def _send_email(self, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Send an email notification.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body (HTML)
            
        Returns:
            Delivery details
        """
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = to_email
        
        # Attach HTML version
        html_part = MIMEText(body, "html")
        msg.attach(html_part)
        
        # In production, use proper SMTP with connection pooling
        # This is a simplified version
        try:
            # Use a loop to run SMTP in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_email_sync,
                to_email,
                msg
            )
            
            return {
                "to": to_email,
                "subject": subject,
                "sent_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            raise
    
    def _send_email_sync(self, to_email: str, msg: MIMEMultipart):
        """
        Synchronous email sending (runs in thread pool).
        
        Args:
            to_email: Recipient email
            msg: Email message object
        """
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.smtp_user and self.smtp_password:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
    
    async def _send_sms(self, to_phone: str, message: str) -> Dict[str, Any]:
        """
        Send an SMS notification.
        
        Args:
            to_phone: Recipient phone number
            message: SMS text
            
        Returns:
            Delivery details
        """
        # In production, integrate with SMS provider like Twilio
        # This is a placeholder implementation
        
        # Simulate API call
        await asyncio.sleep(0.1)
        
        logger.info(f"SMS sent to {to_phone}: {message[:50]}...")
        
        return {
            "to": to_phone,
            "message_preview": message[:50],
            "sent_at": datetime.utcnow().isoformat()
        }
    
    async def _send_in_app(
        self,
        user_id: str,
        template_name: str,
        template_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send an in-app notification.
        
        Args:
            user_id: Recipient user ID
            template_name: Template name
            template_data: Template data
            
        Returns:
            Notification details
        """
        notification = {
            "id": str(uuid4()),
            "user_id": user_id,
            "type": template_name,
            "data": template_data,
            "created_at": datetime.utcnow().isoformat(),
            "read": False,
            "archived": False
        }
        
        # Store in Redis for quick access
        if self.redis_client:
            await self.redis_client.lpush(
                f"in_app_notifications:{user_id}",
                json.dumps(notification)
            )
            await self.redis_client.ltrim(
                f"in_app_notifications:{user_id}",
                0,
                49  # Keep last 50 notifications
            )
        
        return notification
    
    async def _send_webhook(self, webhook_url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a webhook notification.
        
        Args:
            webhook_url: Webhook endpoint URL
            data: Payload data
            
        Returns:
            Delivery details
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    webhook_url,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.webhook_timeout)
                ) as response:
                    response_text = await response.text()
                    
                    return {
                        "url": webhook_url,
                        "status_code": response.status,
                        "response": response_text[:200],
                        "sent_at": datetime.utcnow().isoformat()
                    }
                    
            except asyncio.TimeoutError:
                logger.error(f"Webhook timeout for {webhook_url}")
                raise Exception("Webhook timeout")
            except Exception as e:
                logger.error(f"Webhook failed for {webhook_url}: {str(e)}")
                raise
    
    async def send_bulk_notification(
        self,
        user_ids: List[str],
        template_name: str,
        template_data: Dict[str, Any],
        channel: str = NotificationChannel.IN_APP
    ) -> List[Dict[str, Any]]:
        """
        Send notification to multiple users.
        
        Args:
            user_ids: List of recipient user IDs
            template_name: Template name
            template_data: Template data
            channel: Notification channel
            
        Returns:
            List of notification results
        """
        tasks = []
        for user_id in user_ids:
            tasks.append(self.send_notification(
                user_id,
                template_name,
                template_data,
                channels=[channel],
                priority=NotificationPriority.LOW
            ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter and log results
        successful = []
        failed = []
        
        for result in results:
            if isinstance(result, Exception):
                failed.append(str(result))
            elif result.get("status") == "delivered":
                successful.append(result)
            else:
                failed.append(result.get("error", "Unknown error"))
        
        logger.info(f"Bulk notification: {len(successful)} successful, {len(failed)} failed")
        
        return [r for r in results if not isinstance(r, Exception)]
    
    async def get_user_notifications(
        self,
        user_id: str,
        limit: int = 20,
        include_read: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get in-app notifications for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of notifications
            include_read: Whether to include read notifications
            
        Returns:
            List of notifications
        """
        notifications = []
        
        if self.redis_client:
            # Get notifications from Redis
            raw_notifications = await self.redis_client.lrange(
                f"in_app_notifications:{user_id}",
                0,
                limit - 1
            )
            
            for raw in raw_notifications:
                notification = json.loads(raw)
                if include_read or not notification.get("read"):
                    notifications.append(notification)
        
        return notifications
    
    async def mark_notification_read(self, user_id: str, notification_id: str) -> bool:
        """
        Mark an in-app notification as read.
        
        Args:
            user_id: User ID
            notification_id: Notification ID
            
        Returns:
            True if successful
        """
        if not self.redis_client:
            return False
        
        # Get all notifications
        notifications = await self.redis_client.lrange(
            f"in_app_notifications:{user_id}",
            0,
            -1
        )
        
        # Find and update the notification
        updated = False
        for i, raw in enumerate(notifications):
            notification = json.loads(raw)
            if notification["id"] == notification_id:
                notification["read"] = True
                notification["read_at"] = datetime.utcnow().isoformat()
                await self.redis_client.lset(
                    f"in_app_notifications:{user_id}",
                    i,
                    json.dumps(notification)
                )
                updated = True
                break
        
        return updated
    
    async def send_escalation_notification(
        self,
        query: str,
        user_name: str,
        suggested_agent: str,
        confidence: float,
        escalation_url: str
    ) -> Dict[str, Any]:
        """
        Send a notification for human escalation.
        
        Args:
            query: Original user query
            user_name: Name of the user
            suggested_agent: Suggested agent name
            confidence: Confidence score
            escalation_url: URL for review
            
        Returns:
            Notification result
        """
        template_data = {
            "user_name": user_name,
            "query": query,
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "suggested_agent": suggested_agent,
            "confidence": round(confidence * 100, 1),
            "escalation_url": escalation_url
        }
        
        # Send to all supervisors (in production, get from database)
        supervisor_ids = ["supervisor_1", "supervisor_2"]
        
        results = await self.send_bulk_notification(
            supervisor_ids,
            "human_escalation",
            template_data,
            channel=NotificationChannel.EMAIL
        )
        
        return {
            "escalation_id": str(uuid4()),
            "notifications_sent": len(results),
            "results": results
        }


# Helper function to generate UUID
def uuid4():
    """Generate a random UUID."""
    import uuid
    return str(uuid.uuid4())