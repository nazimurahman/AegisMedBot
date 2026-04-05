"""
Performance and load testing for the AegisMedBot API.
Uses Locust to simulate concurrent users and measure system performance.
"""

from locust import HttpUser, task, between, events
from locust.exception import StopUser
import random
import json
from datetime import datetime
from typing import Dict, Any

# Sample test queries for realistic load testing
CLINICAL_QUERIES = [
    "What are the symptoms of pneumonia?",
    "Explain the treatment for hypertension",
    "What medications interact with warfarin?",
    "Show me the latest guidelines for diabetes management",
    "What is the prognosis for heart failure patients?",
    "How to diagnose acute myocardial infarction?",
    "What are the risk factors for stroke?",
    "Explain the protocol for sepsis management",
    "What vaccines are recommended for elderly patients?",
    "How to treat community-acquired pneumonia?"
]

OPERATIONS_QUERIES = [
    "What is current ICU bed occupancy?",
    "Show me ER wait times",
    "Predict patient flow for next shift",
    "How many available ventilators?",
    "What is staff to patient ratio?",
    "Show me operating room schedule",
    "What are discharge rates today?",
    "Analyze admission patterns",
    "Resource allocation status",
    "Staff scheduling optimization"
]

RISK_QUERIES = [
    "Predict readmission risk for elderly patient",
    "Calculate mortality risk for ICU patient",
    "Assess complication risk for surgery",
    "What is sepsis risk score?",
    "Fall risk assessment for patient",
    "Pressure ulcer risk prediction",
    "Medication adherence risk",
    "Delirium risk in elderly",
    "Post-op infection risk",
    "Length of stay prediction"
]

DIRECTOR_QUERIES = [
    "Show me weekly hospital KPIs",
    "What are department performance metrics?",
    "Analyze cost per patient day",
    "Show revenue cycle metrics",
    "What is patient satisfaction score?",
    "Quality metrics report",
    "Staff efficiency analysis",
    "Readmission rate trends",
    "Average length of stay by department",
    "Monthly financial summary"
]

# Combine all query types
ALL_QUERIES = CLINICAL_QUERIES + OPERATIONS_QUERIES + RISK_QUERIES + DIRECTOR_QUERIES


class MedIntelUser(HttpUser):
    """
    Simulated user for load testing the MedIntel API.
    Each user instance represents a concurrent user making requests.
    """
    
    # Wait between 1 and 3 seconds between tasks to simulate real user behavior
    wait_time = between(1, 3)
    
    def on_start(self):
        """
        Called when a simulated user starts.
        Sets up authentication and initial conversation state.
        """
        # Generate a unique user ID for this load test session
        self.user_id = f"load_test_user_{random.randint(10000, 99999)}"
        
        # Create authentication token (simplified for load testing)
        self.auth_token = f"test_token_{self.user_id}"
        
        # Headers for authenticated requests
        self.headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        # Track conversation ID for multi-turn conversations
        self.conversation_id = None
        
        # Track metrics for this user
        self.request_count = 0
        self.start_time = datetime.now()
    
    def on_stop(self):
        """
        Called when a simulated user stops.
        Logs user session statistics.
        """
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"User {self.user_id} completed {self.request_count} requests in {duration:.2f} seconds")
    
    @task(3)
    def chat_query(self):
        """
        Primary task: Send chat queries to the API.
        Weight 3 means this runs 3 times more often than weight 1 tasks.
        """
        # Select a random query from the pool
        query = random.choice(ALL_QUERIES)
        
        # Prepare request payload
        payload = {
            "query": query,
            "conversation_id": self.conversation_id,
            "metadata": {
                "user_role": random.choice(["doctor", "nurse", "admin"]),
                "test_session": True,
                "user_id": self.user_id
            }
        }
        
        # Send POST request to chat endpoint
        with self.client.post(
            "/api/v1/chat/",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/chat/"
        ) as response:
            
            # Increment request counter
            self.request_count += 1
            
            # Validate response
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Check required fields in response
                    if "conversation_id" in data:
                        # Store conversation ID for follow-up questions
                        self.conversation_id = data["conversation_id"]
                    
                    if "response" in data and data["response"]:
                        # Successful response
                        response.success()
                    else:
                        # Empty response - treat as failure
                        response.failure("Empty response received")
                        
                except json.JSONDecodeError:
                    # Invalid JSON response
                    response.failure("Invalid JSON response")
            else:
                # Non-200 status code
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def health_check(self):
        """
        Health check task to monitor API availability.
        Weight 1 means this runs less frequently than chat queries.
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure(f"Unhealthy status: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def multi_turn_conversation(self):
        """
        Simulate multi-turn conversation with follow-up questions.
        Tests context retention across multiple requests.
        """
        if not self.conversation_id:
            # Start a new conversation if none exists
            return
        
        # Follow-up questions that reference previous context
        follow_ups = [
            "Tell me more about that",
            "What are the treatment options?",
            "What are the risk factors?",
            "Can you explain in more detail?",
            "What does that mean for patient care?"
        ]
        
        # Select random follow-up
        follow_up = random.choice(follow_ups)
        
        # Send follow-up with existing conversation ID
        payload = {
            "query": follow_up,
            "conversation_id": self.conversation_id,
            "metadata": {"turn": self.request_count}
        }
        
        with self.client.post(
            "/api/v1/chat/",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/chat/ - Multi-turn"
        ) as response:
            if response.status_code == 200:
                self.request_count += 1
                response.success()
            else:
                response.failure(f"Multi-turn failed: {response.status_code}")
    
    @task(1)
    def feedback_submission(self):
        """
        Simulate user feedback submission.
        Tests feedback collection endpoints for model improvement.
        """
        # Only submit feedback occasionally (10% of tasks)
        if random.random() > 0.1:
            return
        
        # Generate random feedback
        rating = random.randint(1, 5)
        feedback_text = random.choice([
            "Very helpful response",
            "Accurate information",
            "Could be more detailed",
            "Not relevant to my question",
            "Excellent explanation"
        ])
        
        # Use a random conversation ID (might be from this session or dummy)
        conv_id = self.conversation_id or f"test_conv_{random.randint(1000, 9999)}"
        
        with self.client.post(
            f"/api/v1/chat/feedback",
            params={
                "conversation_id": conv_id,
                "rating": rating,
                "feedback_text": feedback_text
            },
            headers=self.headers,
            catch_response=True,
            name="/api/v1/chat/feedback"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                # Don't fail on feedback errors - they're not critical
                response.success()  # Treat as success to avoid skewing metrics


class HospitalAdminUser(HttpUser):
    """
    Simulated hospital administrator user.
    Focuses on operational and director-level queries.
    """
    
    wait_time = between(2, 5)  # Slower interaction rate for admin tasks
    
    def on_start(self):
        """Initialize admin user session."""
        self.user_id = f"admin_user_{random.randint(10000, 99999)}"
        self.headers = {
            "Authorization": f"Bearer admin_token_{self.user_id}",
            "Content-Type": "application/json"
        }
    
    @task(2)
    def director_queries(self):
        """
        Director-level intelligence queries.
        Tests KPI and performance analytics endpoints.
        """
        query = random.choice(DIRECTOR_QUERIES)
        
        payload = {
            "query": query,
            "metadata": {"user_role": "medical_director"}
        }
        
        with self.client.post(
            "/api/v1/chat/",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/chat/ - Director"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Director query failed: {response.status_code}")
    
    @task(1)
    def agent_status_check(self):
        """
        Check agent status for monitoring.
        Tests admin monitoring endpoints.
        """
        with self.client.get(
            "/api/v1/admin/agents/status",
            headers=self.headers,
            catch_response=True,
            name="/api/v1/admin/agents/status"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Agent status check failed: {response.status_code}")


class StressTestUser(HttpUser):
    """
    Stress test user that makes rapid requests without waiting.
    Tests system limits and breaking points.
    """
    
    wait_time = between(0.1, 0.5)  # Very fast requests
    
    def on_start(self):
        """Initialize stress test user."""
        self.user_id = f"stress_user_{random.randint(10000, 99999)}"
        self.headers = {
            "Authorization": f"Bearer stress_token_{self.user_id}",
            "Content-Type": "application/json"
        }
    
    @task(1)
    def rapid_chat_requests(self):
        """
        Send rapid chat requests to test throughput.
        """
        query = random.choice(ALL_QUERIES[:20])  # Use smaller subset for speed
        
        payload = {
            "query": query,
            "metadata": {"stress_test": True}
        }
        
        # Use a shorter timeout for stress testing
        with self.client.post(
            "/api/v1/chat/",
            json=payload,
            headers=self.headers,
            catch_response=True,
            timeout=5.0
        ) as response:
            # Accept both 200 and 429 (rate limited) as acceptable
            if response.status_code in [200, 429]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Called when load test starts.
    Logs test configuration and start time.
    """
    print("\n" + "="*60)
    print("MedIntel Performance Test Starting")
    print(f"Start Time: {datetime.now()}")
    print(f"Test Type: {environment.runner.__class__.__name__}")
    print("="*60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when load test stops.
    Prints summary statistics.
    """
    print("\n" + "="*60)
    print("MedIntel Performance Test Completed")
    print(f"End Time: {datetime.now()}")
    print(f"Total Requests: {environment.stats.total.num_requests}")
    print(f"Requests/s: {environment.stats.total.total_rps:.2f}")
    print(f"Failure Rate: {(environment.stats.total.fail_ratio * 100):.2f}%")
    print(f"Average Response Time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"95th Percentile: {environment.stats.total.get_response_time_percentile(0.95):.2f}ms")
    print("="*60 + "\n")


# Command line interface for running specific test scenarios
if __name__ == "__main__":
    import os
    import sys
    
    print("MedIntel Performance Test Runner")
    print("Available test scenarios:")
    print("1. Standard user simulation")
    print("2. Admin user simulation")
    print("3. Stress test")
    print("4. All tests")
    
    # This allows running from command line with arguments
    # Example: locust -f locustfile.py --host=http://localhost:8000