# Runbook: Configure Azure Key Vault for ELSPETH Secrets

Set up Azure Key Vault as the secret store for an ELSPETH pipeline.

---

## Symptoms

You need to:
- Load API keys and tokens from Azure Key Vault instead of environment variables
- Ensure secrets are not logged or committed to version control
- Audit which secrets were accessed and when
- Use the same secrets across multiple pipeline runs and environments

---

## Prerequisites

- Azure subscription with permissions to create resources
- `az` CLI installed and logged in (`az login`)
- A resource group where you can create a Key Vault
- User or service principal with contributor access to the resource group
- For production: Managed Identity configured (if running on Azure VMs, App Service, AKS)

**Check CLI access:**

```bash
az account show
# Should show your subscription info

az group list --query "[].{name:name,id:id}" | head -5
# List available resource groups
```

---

## Step 1: Create Azure Key Vault

Create a Key Vault instance with appropriate access controls.

### 1.1 Create the Vault

```bash
# Set variables
VAULT_NAME="elspeth-prod-vault"          # Must be globally unique (3-24 chars, alphanumeric + hyphen)
RESOURCE_GROUP="my-resource-group"
LOCATION="eastus"

# Create the vault
az keyvault create \
  --name "$VAULT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --enable-rbac-authorization
```

The `--enable-rbac-authorization` flag is recommended for production. It uses Azure RBAC (Role-Based Access Control) instead of vault policies.

### 1.2 Verify the Vault

```bash
# Get the vault URL
VAULT_URL=$(az keyvault show \
  --name "$VAULT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.vaultUri \
  --output tsv)

echo "Vault URL: $VAULT_URL"
# Output: https://elspeth-prod-vault.vault.azure.net/
```

---

## Step 2: Grant Access to the Vault

Give your user and/or service principal permission to read secrets.

### 2.1 Grant Access to Your User Account

```bash
# Get your Azure user object ID
USER_OBJECT_ID=$(az ad signed-in-user show --query id -o tsv)

# Assign the "Key Vault Secrets Officer" role
# This allows reading, writing, and managing secrets
az role assignment create \
  --role "Key Vault Secrets Officer" \
  --assignee-object-id "$USER_OBJECT_ID" \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$VAULT_NAME"
```

### 2.2 Grant Access to a Service Principal (for production)

If your pipeline runs on an Azure-managed service (VM, App Service, AKS), grant the Managed Identity access:

```bash
# For App Service
SERVICE_PRINCIPAL_ID=$(az webapp identity show \
  --name my-app-service \
  --resource-group "$RESOURCE_GROUP" \
  --query principalId \
  -o tsv)

# For AKS pod with managed identity
SERVICE_PRINCIPAL_ID=$(az aks show \
  --name my-cluster \
  --resource-group "$RESOURCE_GROUP" \
  --query identityProfile.kubeletidentity.objectId \
  -o tsv)

# Assign the "Key Vault Secrets User" role (read-only)
az role assignment create \
  --role "Key Vault Secrets User" \
  --assignee-object-id "$SERVICE_PRINCIPAL_ID" \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$VAULT_NAME"
```

### 2.3 Verify Access

```bash
# Test that you can access the vault (should not error)
az keyvault secret list --vault-name "$VAULT_NAME"
```

---

## Step 3: Create Secrets in Key Vault

Add the secrets your pipeline needs.

### 3.1 Common Secrets

```bash
# Azure OpenAI API key
az keyvault secret set \
  --vault-name "$VAULT_NAME" \
  --name "azure-openai-key" \
  --value "your-actual-api-key-here"

# Azure OpenAI endpoint
az keyvault secret set \
  --vault-name "$VAULT_NAME" \
  --name "openai-endpoint" \
  --value "https://your-openai-instance.openai.azure.com/"

# ELSPETH fingerprint key (used for secret fingerprinting in audit trail)
az keyvault secret set \
  --vault-name "$VAULT_NAME" \
  --name "elspeth-fingerprint-key" \
  --value "$(openssl rand -hex 32)"  # Generate a random 64-char hex string
```

### 3.2 Verify Secrets

```bash
# List all secret names (not values)
az keyvault secret list --vault-name "$VAULT_NAME" --query "[].name"

# Verify a specific secret exists without printing its value
az keyvault secret show \
  --vault-name "$VAULT_NAME" \
  --name "azure-openai-key" \
  --query id \
  --output tsv
```

### 3.3 Rotate Secrets (Later)

When you need to update a secret:

```bash
az keyvault secret set \
  --vault-name "$VAULT_NAME" \
  --name "azure-openai-key" \
  --value "new-api-key-value"

# The new value is immediately available to pipelines
# Old runs with cached credentials will continue working until they expire
```

---

## Step 4: Configure Your Pipeline

Update your ELSPETH pipeline settings to load secrets from Key Vault.

### 4.1 Create settings.yaml with Secrets Section

```yaml
# settings.yaml

# Top-level pipeline config
source:
  plugin: csv
  options:
    path: input/data.csv

sinks:
  output:
    plugin: csv
    options:
      path: output/results.csv

# Load secrets from Azure Key Vault
secrets:
  source: keyvault
  vault_url: https://elspeth-prod-vault.vault.azure.net
  mapping:
    AZURE_OPENAI_KEY: azure-openai-key
    AZURE_OPENAI_ENDPOINT: openai-endpoint
    ELSPETH_FINGERPRINT_KEY: elspeth-fingerprint-key

# Now you can use ${AZURE_OPENAI_KEY} in other config sections
transforms:
  - plugin: llm
    options:
      provider: azure
      api_key: ${AZURE_OPENAI_KEY}
      endpoint: ${AZURE_OPENAI_ENDPOINT}
      model: gpt-4
      queries:
        - "Classify this record: ${category}"
```

### 4.2 Important: Vault URL Must Be Literal

The `vault_url` field MUST be a literal HTTPS URL. Environment variable references are NOT supported:

```yaml
# WRONG - This will fail
secrets:
  source: keyvault
  vault_url: ${AZURE_KEYVAULT_URL}  # ERROR: Variables not expanded here
  mapping: ...

# CORRECT - Literal URL
secrets:
  source: keyvault
  vault_url: https://elspeth-prod-vault.vault.azure.net
  mapping: ...
```

### 4.3 Vault URL Is Restricted to Approved Azure Key Vault Endpoints

For security, `vault_url` is validated as a real Azure Key Vault endpoint at
config load — before ELSPETH ever authenticates to it. This prevents a settings
file from aiming ELSPETH's Azure credentials at an arbitrary host. The URL must:

- use HTTPS;
- carry no userinfo (`user:pass@`), no path, no query string, and no fragment;
- use no port other than 443;
- have a host ending in an approved Azure Key Vault suffix — `.vault.azure.net`
  (public cloud), `.vault.azure.cn` (China), `.vault.usgovcloudapi.net`
  (US Government), or `.vault.microsoftazure.de` (legacy Germany). Private Link
  keeps the public `<vault>.vault.azure.net` host, so it is already covered.

Anything else is rejected before a credentialed request is made.

#### Optional: pin to exact vaults (deployment allowlist)

The suffix check still permits *any* Azure Key Vault, including one in another
tenant. To close that gap, a deployment can pin the exact vault URL(s) it trusts
via the `ELSPETH_KEYVAULT_ALLOWED_VAULT_URLS` environment variable — set by the
deployment/operator, **never** in pipeline YAML:

```bash
# Comma- or space-separated list of exact vault URLs
export ELSPETH_KEYVAULT_ALLOWED_VAULT_URLS="https://elspeth-prod-vault.vault.azure.net"
```

When set, a `vault_url` that is not an exact match is refused before any Key
Vault call. When unset, the suffix check above is the floor.

Why? Secrets must be loaded before Dynaconf resolves `${VAR}` syntax. The vault URL is needed to fetch the secrets in the first place.

### 4.3 Mapping Format

The `mapping` section maps environment variable names to Key Vault secret names:

```yaml
mapping:
  AZURE_OPENAI_KEY: azure-openai-key           # Env var name -> Vault secret name
  AZURE_OPENAI_ENDPOINT: openai-endpoint
  ELSPETH_FINGERPRINT_KEY: elspeth-fingerprint-key
```

When the pipeline starts:
1. ELSPETH connects to Key Vault using DefaultAzureCredential
2. Fetches the secret value for each entry in `mapping`
3. Injects into environment (e.g., `AZURE_OPENAI_KEY=<secret-value>`)
4. Then Dynaconf resolves `${AZURE_OPENAI_KEY}` to the secret value

---

## Step 5: Verify Setup with Dry-Run

Test that everything works before running the full pipeline.

### 5.1 Run Validation

```bash
# Validate the config file format
elspeth validate --settings settings.yaml

# Should output: "Configuration is valid"
```

### 5.2 Test Secret Loading (Dry-Run)

```bash
# Load config and secrets without executing any transforms
# This tests authentication and secret resolution
elspeth run \
  --settings settings.yaml \
  --dry-run

# Should complete without errors
# If successful, you'll see:
# - "Loading secrets from Key Vault"
# - "Loaded N secrets"
# - "DAG validation passed"
# - "Dry-run completed successfully"
```

### 5.3 Check Secret Resolution Audit Trail

After a successful run, secrets are recorded in the audit trail:

```bash
# Query the audit database for secret resolutions
sqlite3 runs/audit.db "
  SELECT
    secret_name,
    fingerprint,
    resolution_latency_ms,
    resolved_at
  FROM secret_resolutions
  ORDER BY resolved_at DESC
  LIMIT 10;
"

# Output shows:
# - secret_name: The vault secret name (e.g., 'azure-openai-key')
# - fingerprint: HMAC fingerprint of the secret value (not the value itself)
# - resolution_latency_ms: Time to fetch from Key Vault (useful for performance debugging)
# - resolved_at: When the secret was loaded
```

The fingerprint allows auditors to verify which secret was used without storing the actual secret value.

### 5.4 Full Pipeline Test

```bash
# Run the actual pipeline
elspeth run --settings settings.yaml --execute

# Monitor progress
# If successful: all rows processed, no auth errors
# If failed: see Troubleshooting section below
```

---

## Troubleshooting

### Error: "DefaultAzureCredential could not find valid credentials"

**Symptom:** Pipeline fails during secret loading with authentication error.

**Causes & Solutions:**

1. **Not logged in to Azure CLI:**
   ```bash
   az login
   # Complete the browser-based login flow
   ```

2. **Using service principal with wrong credentials:**
   ```bash
   export AZURE_CLIENT_ID="your-client-id"
   export AZURE_CLIENT_SECRET="your-client-secret"
   export AZURE_TENANT_ID="your-tenant-id"

   # Verify credentials
   az account show --query tenantId
   ```

3. **Running on Azure VM without Managed Identity:**
   - If using App Service: Configure Managed Identity in Settings → Identity
   - If using AKS: Enable workload identity and bind it to the pod
   - If using VM: Assign a Managed Identity in the Azure portal

4. **Credential cache expired:**
   ```bash
   az logout && az login
   ```

### Error: "Secret 'XXX' not found in Key Vault"

**Symptom:** Pipeline fails with "Secret not found" error for a specific secret name.

**Causes & Solutions:**

1. **Wrong secret name in mapping:**
   ```yaml
   # Check what secrets exist in your vault
   # (Wrong name in mapping)
   mapping:
     AZURE_OPENAI_KEY: azure-openai-kee  # TYPO!

   # Correct:
   mapping:
     AZURE_OPENAI_KEY: azure-openai-key
   ```

2. **Secret doesn't exist in vault:**
   ```bash
   # List all secrets to verify
   az keyvault secret list \
     --vault-name "elspeth-prod-vault" \
     --query "[].name"

   # If missing, create it
   az keyvault secret set \
     --vault-name "elspeth-prod-vault" \
     --name "azure-openai-key" \
     --value "your-api-key-here"
   ```

3. **Wrong vault name in settings:**
   ```yaml
   # Wrong vault URL
   vault_url: https://elspeth-dev-vault.vault.azure.net

   # Correct (matches where you created the vault)
   vault_url: https://elspeth-prod-vault.vault.azure.net
   ```

### Error: "Forbidden" or "Access Denied"

**Symptom:** Authentication succeeds but secret fetch fails with permission error.

**Causes & Solutions:**

1. **Missing RBAC role assignment:**
   ```bash
   # Check your role assignment
   az role assignment list \
     --assignee "your-email@example.com" \
     --resource-group "my-resource-group" \
     --query "[].roleDefinitionName"

   # Should show "Key Vault Secrets Officer" or "Key Vault Secrets User"

   # If missing, grant access (see Step 2 above)
   ```

2. **Service principal is not in the correct scope:**
   ```bash
   # If using Managed Identity on App Service:
   az webapp identity show --name my-app --resource-group my-group

   # Copy the principalId and create role assignment to the vault
   az role assignment create \
     --role "Key Vault Secrets User" \
     --assignee-object-id "<principalId>" \
     --scope "/subscriptions/.../vaults/elspeth-prod-vault"
   ```

3. **RBAC changes need time to propagate:**
   ```bash
   # Wait 1-2 minutes for Azure RBAC to propagate
   # Then retry
   sleep 60
   elspeth run --settings settings.yaml --execute
   ```

### Error: "vault_url cannot contain ${VAR}"

**Symptom:** Configuration error when vault_url uses environment variable syntax.

**Causes & Solutions:**

```yaml
# WRONG - Causes error
secrets:
  source: keyvault
  vault_url: ${AZURE_KEYVAULT_URL}  # Not allowed!
  mapping: ...

# CORRECT - Use literal URL
secrets:
  source: keyvault
  vault_url: https://elspeth-prod-vault.vault.azure.net
  mapping: ...
```

**Why?** The `vault_url` is used to load secrets before Dynaconf resolves variables. At that point, the environment variable doesn't exist yet (it depends on loading the secret first).

**If you need dynamic vault URLs:**

Option 1: Use a literal URL per environment:
```yaml
# dev-settings.yaml
vault_url: https://elspeth-dev-vault.vault.azure.net

# prod-settings.yaml
vault_url: https://elspeth-prod-vault.vault.azure.net
```

Option 2: Use environment variable BEFORE ELSPETH starts:
```bash
# Shell script
VAULT_NAME="elspeth-${ENVIRONMENT}-vault"
VAULT_URL="https://${VAULT_NAME}.vault.azure.net"

# Dynamically generate settings.yaml with the URL
sed "s|VAULT_URL_PLACEHOLDER|$VAULT_URL|g" settings.template.yaml > settings.yaml

# Then run pipeline
elspeth run --settings settings.yaml --execute
```

### Error: "ELSPETH_FINGERPRINT_KEY is required when loading secrets from Key Vault"

**Symptom:** Pipeline fails before loading Key Vault secrets because no fingerprint key is available.

**Why:** Key Vault secret loading records audit fingerprints for every resolved secret. ELSPETH requires `ELSPETH_FINGERPRINT_KEY` either in the environment before startup or in the Key Vault mapping.

**Solution:**

```bash
# Create a random fingerprint key
az keyvault secret set \
  --vault-name "elspeth-prod-vault" \
  --name "elspeth-fingerprint-key" \
  --value "$(openssl rand -hex 32)"

# Or skip the fingerprint key (if not using audit fingerprints):
# Set it in the process environment before ELSPETH starts
export ELSPETH_FINGERPRINT_KEY="$(openssl rand -hex 32)"

# If you use the environment approach, remove it from the Key Vault mapping:
mapping:
  AZURE_OPENAI_KEY: azure-openai-key
  AZURE_OPENAI_ENDPOINT: openai-endpoint
  # ELSPETH_FINGERPRINT_KEY: elspeth-fingerprint-key  <- Remove this line
```

---

## Migration from Old Approach

If you previously used environment variables for secrets, migrate to Key Vault.

### Before (Environment Variables)

```bash
# Old: secrets stored in .env or exported in shell
export AZURE_OPENAI_KEY="sk-..."
export AZURE_OPENAI_ENDPOINT="https://..."

# .env file (only for local development, and never committed)
AZURE_OPENAI_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://...

# Pipeline config didn't have secrets section
# settings.yaml (old)
transforms:
  - plugin: llm
    options:
      provider: azure
      api_key: ${AZURE_OPENAI_KEY}  # Loaded from environment
```

**Problems with old approach:**
- Secrets can be accidentally committed to git if in .env
- Hard to rotate secrets across multiple pipelines
- No audit trail of which secrets were used when
- Different secrets needed per environment = multiple .env files

### After (Key Vault)

```yaml
# settings.yaml (new)
secrets:
  source: keyvault
  vault_url: https://elspeth-prod-vault.vault.azure.net
  mapping:
    AZURE_OPENAI_KEY: azure-openai-key
    AZURE_OPENAI_ENDPOINT: openai-endpoint

transforms:
  - plugin: llm
    options:
      provider: azure
      api_key: ${AZURE_OPENAI_KEY}  # Same syntax, loaded from vault
      endpoint: ${AZURE_OPENAI_ENDPOINT}
```

**Benefits of new approach:**
- Secret values are fetched at startup and injected into the process environment; plaintext is not stored in ELSPETH's audit records
- RBAC controls who can access each secret
- Audit trail shows when/which secrets were accessed
- Same settings.yaml across all environments (vault URLs differ)
- Rotate secrets in vault = instantly updated for all pipelines

### Migration Steps

1. **Create Key Vault** (Step 1 above)

2. **Grant access to your user/service** (Step 2 above)

3. **Migrate secrets one by one:**
   ```bash
   # Get the current value through your approved secret handoff path.
   # Do not print it to the terminal or commit it to a file.
   read -rsp "Current AZURE_OPENAI_KEY: " CURRENT_VALUE
   echo

   # Store in Key Vault
   az keyvault secret set \
     --vault-name "elspeth-prod-vault" \
     --name "azure-openai-key" \
     --value "$CURRENT_VALUE"

   # Verify it was stored without printing the value
   az keyvault secret show \
     --vault-name "elspeth-prod-vault" \
     --name "azure-openai-key" \
     --query id --output tsv
   ```

4. **Update settings.yaml** to add `secrets:` section (Step 4 above)

5. **Test with dry-run** (Step 5.2 above)

6. **Remove .env file** from git (if not already done):
   ```bash
   git rm --cached .env
   echo ".env" >> .gitignore
   git add .gitignore
   git commit -m "Remove .env from git - secrets now in Key Vault"
   ```

7. **Update deployment instructions:**
   - Old: "Set up .env file with secrets"
   - New: "Authenticate with `az login` and grant vault access (Step 2)"

---

## Audit Trail: Secret Resolutions

Every time a pipeline loads secrets from Key Vault, it's recorded in the audit trail.

### Viewing Secret Resolutions

```bash
# Connect to the audit database
sqlite3 runs/audit.db

# Query secret resolution records
SELECT
  run_id,
  secret_name,
  fingerprint,
  resolution_latency_ms,
  resolved_at
FROM secret_resolutions
WHERE run_id = '<RUN_ID>'
ORDER BY resolved_at;
```

### What Gets Recorded

| Field | What It Contains | Why |
|-------|------------------|-----|
| `run_id` | The pipeline run ID | Links secret to the specific run |
| `secret_name` | Name in Key Vault (e.g., 'azure-openai-key') | Logs which secret was needed |
| `fingerprint` | HMAC-SHA256 of secret value | Proves which secret version was used, without storing the value |
| `resolution_latency_ms` | Time to fetch from vault | Detects network/vault performance issues |
| `resolved_at` | Timestamp | Audit trail of access events |

### Why Fingerprints Instead of Values?

For audit integrity, we record a fingerprint (HMAC) of the secret, not the secret itself:

```python
# Never stored in audit trail:
secret_value = "sk-proj-1234567890..."

# Stored instead:
fingerprint = hmac.new(
    fingerprint_key.encode(),
    secret_value.encode(),
    hashlib.sha256
).hexdigest()
# Result: a1b2c3d4e5f6... (deterministic, no way to reverse)
```

**Benefits:**
- Auditors can verify which version of the secret was used
- If a secret is compromised, audit shows when the old value was used
- Secret value is never written to disk/logs

---

## Performance Considerations

### Secret Loading Latency

First secret load can take 100-500ms (Azure SDK initialization). Subsequent loads are much faster (10-50ms) due to caching.

```bash
# Check resolution_latency_ms in audit trail
sqlite3 runs/audit.db "
  SELECT
    secret_name,
    AVG(resolution_latency_ms) as avg_latency,
    MAX(resolution_latency_ms) as max_latency
  FROM secret_resolutions
  GROUP BY secret_name;
"

# Example output:
# secret_name                | avg_latency | max_latency
# azure-openai-key           | 45          | 230
# openai-endpoint            | 15          | 18
# elspeth-fingerprint-key    | 12          | 15
```

### Optimization

For pipelines with many runs, consider:

1. **Reduce secret count:** Only load secrets your pipeline actually uses

   ```yaml
   # Don't add unused secrets to mapping
   mapping:
     AZURE_OPENAI_KEY: azure-openai-key        # Used
     # DATABASE_PASSWORD: db-password           # Commented out (not used)
   ```

2. **Cache in environment:** Pre-load secrets before running multiple pipelines

   ```bash
   # Load once
   export AZURE_OPENAI_KEY=$(az keyvault secret show \
     --vault-name "elspeth-prod-vault" \
     --name "azure-openai-key" \
     --query value --output tsv)

   # Run 100 pipelines - env vars reused, no Key Vault calls
   for i in {1..100}; do
     elspeth run --settings settings-$i.yaml --execute
   done
   ```

---

## See Also

- [Configuration Reference](../reference/configuration.md#secrets-settings) - Complete secrets section documentation
- [Resume Failed Run](resume-failed-run.md) - How to resume if secret loading fails
- [Incident Response](incident-response.md) - Troubleshooting any pipeline issues
