AegisMedBot/
│   
├── frontend/
│   ├── app.py
│   ├── components/
│   │   ├── chat_interface.py
│   │   ├── dashboard.py
│   │   ├── analytics.py
│   │   └── agent_monitor.py
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── requirements.txt
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── agents.py
│   │   │   ├── patients.py
│   │   │   └── admin.py
│   │   ├── middleware/
│   │   │   ├── auth.py
│   │   │   ├── rate_limit.py
│   │   │   └── logging.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   ├── database.py
│   │   └── cache.py
│   ├── models/
│   │   ├── schemas/
│   │   │   ├── patient.py
│   │   │   ├── agent.py
│   │   │   └── response.py
│   │   └── enums.py
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── audit_service.py
│   │   └── notification_service.py
│   ├── main.py
│   └── requirements.txt
├── agents/
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── agent_orchestrator.py
│   │   ├── task_delegator.py
│   │   └── context_manager.py
│   ├── clinical_agent/
│   │   ├── __init__.py
│   │   ├── clinical_agent.py
│   │   ├── tools/
│   │   │   ├── medical_retriever.py
│   │   │   └── drug_interaction.py
│   │   └── prompts.py
│   ├── risk_agent/
│   │   ├── __init__.py
│   │   ├── risk_predictor.py
│   │   ├── models/
│   │   │   ├── lstm_predictor.py
│   │   │   └── transformer_predictor.py
│   │   └── features.py
│   ├── operations_agent/
│   │   ├── __init__.py
│   │   ├── operations_agent.py
│   │   ├── bed_analyzer.py
│   │   └── flow_predictor.py
│   ├── director_agent/
│   │   ├── __init__.py
│   │   ├── director_intelligence.py
│   │   ├── kpi_analyzer.py
│   │   └── report_generator.py
│   ├── compliance_agent/
│   │   ├── __init__.py
│   │   ├── privacy_guardian.py
│   │   ├── phi_detector.py
│   │   └── audit_logger.py
│   ├── research_agent/
│   │   ├── __init__.py
│   │   ├── research_assistant.py
│   │   ├── paper_summarizer.py
│   │   └── literature_retriever.py
│   ├── base_agent.py
│   └── agent_protocol.py
├── rag_system/
│   ├── __init__.py
│   ├── vector_store/
│   │   ├── qdrant_manager.py
│   │   ├── embeddings.py
│   │   └── schema.py
│   ├── retrievers/
│   │   ├── medical_retriever.py
│   │   ├── clinical_retriever.py
│   │   └── hybrid_retriever.py
│   ├── indexers/
│   │   ├── document_indexer.py
│   │   └── medical_text_processor.py
│   └── data_sources/
│       ├── literature/
│       ├── guidelines/
│       └── policies/
├── ml_training/
│   ├── data/
│   │   ├── processors/
│   │   │   ├── clinical_processor.py
│   │   │   └── ehr_processor.py
│   │   └── datasets/
│   │       └── medical_qa_dataset.py
│   ├── models/
│   │   ├── transformer/
│   │   │   ├── medical_transformer.py
│   │   │   └── configuration.py
│   │   ├── lstm/
│   │   │   └── patient_lstm.py
│   │   └── registry.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── configs/
│   │   │   ├── base_config.yaml
│   │   │   └── lora_config.yaml
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── inference/
│   │   ├── model_server.py
│   │   └── quantized_model.py
│   └── requirements.txt
├── inference/
│   ├── __init__.py
│   ├── engine.py
│   ├── cache.py
│   └── optimizations.py
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alerts.yml
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── model_performance.json
│   │   │   ├── system_metrics.json
│   │   │   └── agent_activity.json
│   │   └── datasources.yaml
│   ├── elasticsearch/
│   │   └── logstash.conf
│   └── metrics/
│       ├── model_metrics.py
│       └── system_metrics.py
├── docker/
│   ├── frontend/
│   │   └── Dockerfile
│   ├── backend/
│   │   └── Dockerfile
│   ├── agents/
│   │   └── Dockerfile
│   ├── rag/
│   │   └── Dockerfile
│   ├── nginx/
│   │   └── nginx.conf
│   └── docker-compose.yml
├── kubernetes/
│   ├── namespaces/
│   │   └── medintel.yaml
│   ├── deployments/
│   │   ├── frontend.yaml
│   │   ├── backend.yaml
│   │   ├── agents.yaml
│   │   ├── rag.yaml
│   │   ├── postgres.yaml
│   │   ├── qdrant.yaml
│   │   └── redis.yaml
│   ├── services/
│   │   ├── frontend-service.yaml
│   │   ├── backend-service.yaml
│   │   ├── agents-service.yaml
│   │   ├── rag-service.yaml
│   │   ├── postgres-service.yaml
│   │   ├── qdrant-service.yaml
│   │   └── redis-service.yaml
│   ├── ingress/
│   │   └── medintel-ingress.yaml
│   ├── configmaps/
│   │   ├── app-config.yaml
│   │   └── agent-config.yaml
│   ├── secrets/
│   │   └── db-secrets.yaml
│   ├── hpa/
│   │   ├── backend-hpa.yaml
│   │   └── agents-hpa.yaml
│   └── storage/
│       ├── postgres-pvc.yaml
│       └── qdrant-pvc.yaml
├── database/
│   ├── migrations/
│   │   ├── versions/
│   │   └── env.py
│   ├── models/
│   │   ├── patient.py
│   │   ├── admission.py
│   │   ├── treatment.py
│   │   ├── staff.py
│   │   └── resource.py
│   ├── repositories/
│   │   ├── patient_repo.py
│   │   ├── admission_repo.py
│   │   └── audit_repo.py
│   └── alembic.ini
├── tests/
│   ├── unit/
│   │   ├── test_agents.py
│   │   ├── test_api.py
│   │   └── test_rag.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── performance/
│       └── locustfile.py
├── scripts/
│   ├── setup/
│   │   ├── init_db.py
│   │   └── load_sample_data.py
│   ├── training/
│   │   ├── train_transformer.py
│   │   └── train_lstm.py
│   └── monitoring/
│       └── setup_grafana.sh
├── docs/
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── agent_design.md
│   │   └── data_flow.md
│   ├── api/
│   │   ├── openapi.yaml
│   │   └── examples.md
│   ├── deployment/
│   │   ├── kubernetes.md
│   │   └── docker.md
│   └── development/
│       └── setup.md
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
├── poetry.lock
└── README.md