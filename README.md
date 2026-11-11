# Azure AI Service (Month 1 Lab)

FastAPI + Azure OpenAI wrapper with 429 retry (`Retry-After`, exponential backoff with jitter),
content-filter handling, `pydantic-settings` config, and structured JSON logging with `request_id`.

## Project Structure

```text
azure-ai-search/
  src/app/
    main.py            # FastAPI app, routers & middleware
    config.py          # pydantic-settings, env-based
    schemas.py         # Pydantic request/response models
    services/
      llm.py           # Azure OpenAI client wrapper
    logging_conf.py    # structured JSON logging
  scripts/
    loadtest_429.py    # 429 concurrency & retry benchmark
  tests/
    test_llm.py        # LLMClient unit tests
    test_main.py       # API endpoint & logging tests
  pyproject.toml       # ruff, mypy, pytest config
  .github/workflows/ci.yml
  Dockerfile
```

## 1. Run Locally (Python venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # fill in AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
uvicorn app.main:app --reload --app-dir src
```

## 2. Run Locally with Docker

```bash
# 1. Prepare environment file
cp .env.example .env
# Edit .env and fill in AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT

# 2. Build the Docker image
docker build -t azure-ai-service:local .

# 3. Run the container
docker run --rm -p 8000:8000 --env-file .env azure-ai-service:local
```

## 3. Test Endpoints

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: demo-001' \
  -d '{"message":"hi"}'
```

## 4. Quality Gates (same as CI)

```bash
ruff check . && ruff format --check . && mypy src && pytest
```

## 5. 429 Experiment

```bash
python scripts/loadtest_429.py --no-retry --concurrency 5 10 20 40
python scripts/loadtest_429.py --retries 5 --concurrency 5 10 20 40
```
Record results in `LAB_NOTES.md`.

## 6. Deploy to Azure (Azure Container Apps)

```bash
az login

RESOURCE_GROUP="rg-azure-ai-lab"
LOCATION="eastus"
ACR_NAME="acrazureailab$RANDOM"
ACA_ENV="env-azure-ai-lab"
APP_NAME="azure-ai-service"

# Create Resource Group & Azure Container Registry
az group create --name $RESOURCE_GROUP --location $LOCATION
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Build & push image remotely in ACR
az acr build --registry $ACR_NAME --image $APP_NAME:v1 .

# Create Container Apps environment
az containerapp env create --name $ACA_ENV --resource-group $RESOURCE_GROUP --location $LOCATION

# Deploy Container App
ACR_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
ACR_USER=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASS=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ACA_ENV \
  --image "$ACR_SERVER/$APP_NAME:v1" \
  --registry-server $ACR_SERVER \
  --registry-username $ACR_USER \
  --registry-password $ACR_PASS \
  --target-port 8000 \
  --ingress external \
  --secrets openai-key="<YOUR_AZURE_OPENAI_API_KEY>" \
  --env-vars \
    AZURE_OPENAI_ENDPOINT="https://<YOUR_RESOURCE>.openai.azure.com/" \
    AZURE_OPENAI_API_KEY=secretref:openai-key \
    AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini" \
    AZURE_OPENAI_API_VERSION="2024-08-01-preview" \
    LOG_LEVEL="INFO"
```
