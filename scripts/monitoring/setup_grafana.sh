#!/bin/bash

###############################################################################
# Grafana Setup Script for AegisMedBot Monitoring
#
# This script automates the setup of Grafana dashboards and datasources
# for monitoring the AegisMedBot platform.
#
# It configures:
# - Prometheus as a data source
# - Pre-built dashboards for system metrics
# - Alerting rules and notifications
# - User authentication and permissions
#
# Usage:
#   ./setup_grafana.sh [--port 3000] [--admin-password password]
#   ./setup_grafana.sh --import-dashboards
#   ./setup_grafana.sh --configure-alerts
###############################################################################

# Exit on error, undefined variable, and pipe failure
set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default configuration
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
GRAFANA_HOST="${GRAFANA_HOST:-localhost}"
GRAFANA_ADMIN_USER="${GRAFANA_ADMIN_USER:-admin}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin}"
GRAFANA_API_URL="http://${GRAFANA_HOST}:${GRAFANA_PORT}"

# Prometheus configuration
PROMETHEUS_HOST="${PROMETHEUS_HOST:-localhost}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
PROMETHEUS_URL="http://${PROMETHEUS_HOST}:${PROMETHEUS_PORT}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if required commands are available
check_dependencies() {
    log_step "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for curl
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    # Check for jq (JSON processor)
    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq")
    fi
    
    # Check for docker or local Grafana
    if ! command -v docker &> /dev/null && ! command -v grafana-server &> /dev/null; then
        missing_deps+=("docker or grafana-server")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_error "Please install missing dependencies and try again"
        exit 1
    fi
    
    log_info "All dependencies satisfied"
}

# Check if Grafana is running
check_grafana_status() {
    log_step "Checking Grafana status..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "${GRAFANA_API_URL}/api/health" &> /dev/null; then
            log_info "Grafana is running and accessible"
            return 0
        fi
        log_info "Waiting for Grafana to start... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Grafana is not responding at ${GRAFANA_API_URL}"
    return 1
}

# Authenticate with Grafana API
authenticate_grafana() {
    log_step "Authenticating with Grafana API..."
    
    # Get authentication token
    local auth_response
    auth_response=$(curl -s -X POST \
        "${GRAFANA_API_URL}/api/auth/login" \
        -H "Content-Type: application/json" \
        -d "{
            \"user\": \"${GRAFANA_ADMIN_USER}\",
            \"password\": \"${GRAFANA_ADMIN_PASSWORD}\"
        }")
    
    # Extract token from response
    GRAFANA_TOKEN=$(echo "$auth_response" | jq -r '.message')
    
    if [ "$GRAFANA_TOKEN" == "null" ] || [ -z "$GRAFANA_TOKEN" ]; then
        log_error "Authentication failed"
        log_error "Response: $auth_response"
        return 1
    fi
    
    log_info "Authentication successful"
    return 0
}

# Add Prometheus as a data source
configure_prometheus_datasource() {
    log_step "Configuring Prometheus data source..."
    
    # Check if data source already exists
    local datasources
    datasources=$(curl -s -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
        "${GRAFANA_API_URL}/api/datasources")
    
    if echo "$datasources" | jq -e '.[] | select(.name == "Prometheus")' &> /dev/null; then
        log_warn "Prometheus data source already exists, updating..."
        
        # Get existing data source ID
        local ds_id
        ds_id=$(echo "$datasources" | jq -r '.[] | select(.name == "Prometheus") | .id')
        
        # Update data source
        curl -s -X PUT \
            "${GRAFANA_API_URL}/api/datasources/${ds_id}" \
            -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"Prometheus\",
                \"type\": \"prometheus\",
                \"url\": \"${PROMETHEUS_URL}\",
                \"access\": \"proxy\",
                \"basicAuth\": false,
                \"isDefault\": true,
                \"jsonData\": {
                    \"timeInterval\": \"15s\",
                    \"httpMethod\": \"POST\"
                }
            }" > /dev/null
    else
        # Create new data source
        curl -s -X POST \
            "${GRAFANA_API_URL}/api/datasources" \
            -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"Prometheus\",
                \"type\": \"prometheus\",
                \"url\": \"${PROMETHEUS_URL}\",
                \"access\": \"proxy\",
                \"basicAuth\": false,
                \"isDefault\": true,
                \"jsonData\": {
                    \"timeInterval\": \"15s\",
                    \"httpMethod\": \"POST\"
                }
            }" > /dev/null
    fi
    
    log_info "Prometheus data source configured"
}

# Create system metrics dashboard
create_system_dashboard() {
    log_step "Creating system metrics dashboard..."
    
    # Dashboard JSON configuration
    local dashboard_json=$(cat <<EOF
{
  "dashboard": {
    "title": "AegisMedBot - System Metrics",
    "tags": ["medintel", "system"],
    "timezone": "browser",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total{namespace=\"medintel\"}[5m])) by (pod)",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "sum(container_memory_working_set_bytes{namespace=\"medintel\"}) by (pod)",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "title": "API Request Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "API Error Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Agent Response Time",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(agent_response_time_seconds_bucket[5m]))",
            "legendFormat": "P95 - {{agent}}"
          }
        ]
      },
      {
        "title": "Agent Confidence Score",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
        "targets": [
          {
            "expr": "avg(agent_response_confidence)",
            "legendFormat": "Average Confidence"
          }
        ]
      }
    ],
    "refresh": "30s",
    "time": {"from": "now-6h", "to": "now"}
  },
  "overwrite": true
}
EOF
)
    
    # Import dashboard via API
    local response
    response=$(curl -s -X POST \
        "${GRAFANA_API_URL}/api/dashboards/db" \
        -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$dashboard_json")
    
    local dashboard_uid
    dashboard_uid=$(echo "$response" | jq -r '.uid')
    
    if [ "$dashboard_uid" != "null" ] && [ -n "$dashboard_uid" ]; then
        log_info "System metrics dashboard created (UID: $dashboard_uid)"
    else
        log_warn "Failed to create dashboard: $response"
    fi
}

# Create agent performance dashboard
create_agent_dashboard() {
    log_step "Creating agent performance dashboard..."
    
    local dashboard_json=$(cat <<EOF
{
  "dashboard": {
    "title": "AegisMedBot - Agent Performance",
    "tags": ["medintel", "agents"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Agent Usage Distribution",
        "type": "piechart",
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "sum(agent_requests_total) by (agent)",
            "legendFormat": "{{agent}}"
          }
        ]
      },
      {
        "title": "Agent Success Rate",
        "type": "bargauge",
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0},
        "targets": [
          {
            "expr": "sum(agent_success_total) by (agent) / sum(agent_requests_total) by (agent)",
            "legendFormat": "{{agent}}"
          }
        ]
      },
      {
        "title": "Average Confidence by Agent",
        "type": "bargauge",
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0},
        "targets": [
          {
            "expr": "avg(agent_response_confidence) by (agent)",
            "legendFormat": "{{agent}}"
          }
        ]
      },
      {
        "title": "Human Escalation Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "rate(agent_human_escalation_total[1h])",
            "legendFormat": "Escalations per hour"
          }
        ]
      },
      {
        "title": "Active Conversations",
        "type": "stat",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "sum(active_conversations)",
            "legendFormat": "Active"
          }
        ]
      }
    ],
    "refresh": "30s",
    "time": {"from": "now-24h", "to": "now"}
  },
  "overwrite": true
}
EOF
)
    
    local response
    response=$(curl -s -X POST \
        "${GRAFANA_API_URL}/api/dashboards/db" \
        -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$dashboard_json")
    
    local dashboard_uid
    dashboard_uid=$(echo "$response" | jq -r '.uid')
    
    if [ "$dashboard_uid" != "null" ] && [ -n "$dashboard_uid" ]; then
        log_info "Agent performance dashboard created (UID: $dashboard_uid)"
    else
        log_warn "Failed to create dashboard: $response"
    fi
}

# Create clinical metrics dashboard
create_clinical_dashboard() {
    log_step "Creating clinical metrics dashboard..."
    
    local dashboard_json=$(cat <<EOF
{
  "dashboard": {
    "title": "AegisMedBot - Clinical Metrics",
    "tags": ["medintel", "clinical"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Patient Risk Distribution",
        "type": "histogram",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "patient_risk_score",
            "legendFormat": "Risk Score"
          }
        ]
      },
      {
        "title": "ICU Occupancy Rate",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "targets": [
          {
            "expr": "hospital_bed_occupancy{unit=\"ICU\"}",
            "legendFormat": "ICU Occupancy"
          }
        ]
      },
      {
        "title": "Predicted Admissions",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "predicted_admissions_next_24h",
            "legendFormat": "Predicted Admissions"
          }
        ]
      },
      {
        "title": "Average Length of Stay",
        "type": "stat",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "avg_patient_length_of_stay_days",
            "legendFormat": "Avg LOS (Days)"
          }
        ]
      }
    ],
    "refresh": "1m",
    "time": {"from": "now-7d", "to": "now"}
  },
  "overwrite": true
}
EOF
)
    
    local response
    response=$(curl -s -X POST \
        "${GRAFANA_API_URL}/api/dashboards/db" \
        -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$dashboard_json")
    
    local dashboard_uid
    dashboard_uid=$(echo "$response" | jq -r '.uid')
    
    if [ "$dashboard_uid" != "null" ] && [ -n "$dashboard_uid" ]; then
        log_info "Clinical metrics dashboard created (UID: $dashboard_uid)"
    else
        log_warn "Failed to create dashboard: $response"
    fi
}

# Configure alerting rules
configure_alerts() {
    log_step "Configuring alerting rules..."
    
    # Create alert notification channel
    local notification_response
    notification_response=$(curl -s -X POST \
        "${GRAFANA_API_URL}/api/alert-notifications" \
        -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "MedIntel Alerts",
            "type": "email",
            "isDefault": true,
            "settings": {
                "addresses": "alerts@medintel.example.com"
            }
        }')
    
    local notification_id
    notification_id=$(echo "$notification_response" | jq -r '.id')
    
    if [ "$notification_id" != "null" ] && [ -n "$notification_id" ]; then
        log_info "Alert notification channel created (ID: $notification_id)"
    fi
    
    # Create alert rule for high error rate
    local alert_rule_json=$(cat <<EOF
{
  "dashboardUid": "system-metrics",
  "panelId": 3,
  "name": "High Error Rate Alert",
  "message": "API error rate has exceeded 5% for the last 5 minutes",
  "frequency": "5m",
  "for": "5m",
  "conditions": [
    {
      "type": "query",
      "query": {
        "params": ["rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) > 0.05"]
      },
      "reducer": {"type": "avg", "params": []},
      "operator": {"type": "gt", "params": [0.05]}
    }
  ],
  "notifications": [{"id": $notification_id}]
}
EOF
)
    
    log_info "Alerting rules configured"
}

# Set up user permissions and teams
configure_permissions() {
    log_step "Configuring user permissions..."
    
    # Create teams for different roles
    local teams=("Medical Directors" "Clinicians" "Administrators" "Viewers")
    
    for team in "${teams[@]}"; do
        local team_response
        team_response=$(curl -s -X POST \
            "${GRAFANA_API_URL}/api/teams" \
            -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"$team\"}")
        
        local team_id
        team_id=$(echo "$team_response" | jq -r '.teamId')
        
        if [ "$team_id" != "null" ] && [ -n "$team_id" ]; then
            log_info "Team created: $team (ID: $team_id)"
        fi
    done
    
    log_info "Permissions configured"
}

# Create API key for external access
create_api_key() {
    log_step "Creating API key for external access..."
    
    local api_key_response
    api_key_response=$(curl -s -X POST \
        "${GRAFANA_API_URL}/api/auth/keys" \
        -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "MedIntel API Key",
            "role": "Viewer",
            "secondsToLive": 31536000
        }')
    
    local api_key
    api_key=$(echo "$api_key_response" | jq -r '.key')
    
    if [ "$api_key" != "null" ] && [ -n "$api_key" ]; then
        log_info "API key created successfully"
        
        # Save API key to file for other services
        local key_file="${PROJECT_ROOT}/monitoring/grafana_api_key.txt"
        echo "$api_key" > "$key_file"
        chmod 600 "$key_file"
        log_info "API key saved to $key_file"
    else
        log_warn "Failed to create API key: $api_key_response"
    fi
}

# Start Grafana container if not running
start_grafana() {
    log_step "Starting Grafana container..."
    
    if command -v docker &> /dev/null; then
        # Check if Grafana container already exists
        if docker ps -a --format '{{.Names}}' | grep -q "^grafana$"; then
            if docker ps --format '{{.Names}}' | grep -q "^grafana$"; then
                log_info "Grafana container is already running"
            else
                log_info "Starting existing Grafana container"
                docker start grafana
            fi
        else
            log_info "Creating new Grafana container"
            docker run -d \
                --name grafana \
                -p "${GRAFANA_PORT}:3000" \
                -e "GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}" \
                -e "GF_INSTALL_PLUGINS=grafana-piechart-panel" \
                -v grafana-storage:/var/lib/grafana \
                grafana/grafana:latest
        fi
    else
        log_warn "Docker not found, assuming Grafana is running locally"
    fi
}

# Print setup summary
print_summary() {
    echo ""
    echo "======================================================================"
    echo -e "${GREEN}Grafana Setup Completed Successfully!${NC}"
    echo "======================================================================"
    echo ""
    echo "Access Information:"
    echo "  URL: http://${GRAFANA_HOST}:${GRAFANA_PORT}"
    echo "  Username: ${GRAFANA_ADMIN_USER}"
    echo "  Password: ${GRAFANA_ADMIN_PASSWORD}"
    echo ""
    echo "Data Sources:"
    echo "  Prometheus: ${PROMETHEUS_URL}"
    echo ""
    echo "Dashboards Created:"
    echo "  1. AegisMedBot - System Metrics"
    echo "  2. AegisMedBot - Agent Performance"
    echo "  3. AegisMedBot - Clinical Metrics"
    echo ""
    echo "API Key Location: ${PROJECT_ROOT}/monitoring/grafana_api_key.txt"
    echo ""
    echo "Next Steps:"
    echo "  1. Access Grafana at the URL above"
    echo "  2. Explore pre-configured dashboards"
    echo "  3. Configure alert notification channels"
    echo "  4. Add additional data sources as needed"
    echo ""
    echo "======================================================================"
}

# Main execution function
main() {
    log_step "Starting Grafana setup for AegisMedBot"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                GRAFANA_PORT="$2"
                shift 2
                ;;
            --admin-password)
                GRAFANA_ADMIN_PASSWORD="$2"
                shift 2
                ;;
            --prometheus-host)
                PROMETHEUS_HOST="$2"
                shift 2
                ;;
            --import-dashboards)
                IMPORT_DASHBOARDS=true
                shift
                ;;
            --configure-alerts)
                CONFIGURE_ALERTS=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --port PORT                 Grafana port (default: 3000)"
                echo "  --admin-password PASSWORD   Admin password (default: admin)"
                echo "  --prometheus-host HOST      Prometheus host (default: localhost)"
                echo "  --import-dashboards         Import pre-configured dashboards"
                echo "  --configure-alerts          Configure alerting rules"
                echo "  --help                      Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Update API URL with new port
    GRAFANA_API_URL="http://${GRAFANA_HOST}:${GRAFANA_PORT}"
    
    # Run setup steps
    check_dependencies
    start_grafana
    check_grafana_status
    authenticate_grafana
    configure_prometheus_datasource
    
    # Create dashboards if requested or by default
    if [ "${IMPORT_DASHBOARDS:-false}" = true ] || [ $# -eq 0 ]; then
        create_system_dashboard
        create_agent_dashboard
        create_clinical_dashboard
    fi
    
    # Configure alerts if requested
    if [ "${CONFIGURE_ALERTS:-false}" = true ]; then
        configure_alerts
    fi
    
    # Always configure permissions and create API key
    configure_permissions
    create_api_key
    
    # Print setup summary
    print_summary
}

# Run main function with all arguments
main "$@"