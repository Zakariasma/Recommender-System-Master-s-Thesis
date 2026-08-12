#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"
VALUES_FILE="${2:-values.yaml}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE"
  exit 1
fi

# Charger les variables
source "$ENV_FILE"

# Vérifications
: "${NAMESPACE:?Missing NAMESPACE in .env}"
: "${SECRET_NAME:?Missing SECRET_NAME in .env}"

# Tableau pour stocker les noms des clés à chiffrer (tout sauf NAMESPACE et SECRET_NAME)
KEYS_TO_ENCRYPT=()

# Lire le fichier .env ligne par ligne pour extraire les clés
while IFS='=' read -r key val; do
  # Ignorer les commentaires et lignes vides
  [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue

  if [[ "$key" != "NAMESPACE" && "$key" != "SECRET_NAME" ]]; then
    KEYS_TO_ENCRYPT+=("$key")
  fi
done < "$ENV_FILE"

if [ ${#KEYS_TO_ENCRYPT[@]} -eq 0 ]; then
  echo "Aucune variable à chiffrer trouvée dans $ENV_FILE"
  exit 1
fi

TMP_CERT="$(mktemp)"
TMP_SECRET="$(mktemp)"
TMP_SEALED="$(mktemp)"

cleanup() { rm -f "$TMP_CERT" "$TMP_SECRET" "$TMP_SEALED"; }
trap cleanup EXIT

echo "Fetching public certificate..."
kubeseal \
  --controller-name=sealed-secrets-controller \
  --controller-namespace=kube-system \
  --fetch-cert > "$TMP_CERT"

echo "Creating temporary Secret with all values..."
# Construire les arguments --from-literal dynamiquement
FROM_LITERAL_ARGS=()
for key in "${KEYS_TO_ENCRYPT[@]}"; do
  # Récupérer la valeur de la variable dynamiquement
  value="${!key}"
  FROM_LITERAL_ARGS+=("--from-literal=${key}=${value}")
done

kubectl create secret generic "$SECRET_NAME" \
  --namespace "$NAMESPACE" \
  --dry-run=client \
  "${FROM_LITERAL_ARGS[@]}" \
  -o yaml > "$TMP_SECRET"

echo "Encrypting all values into one SealedSecret..."
kubeseal \
  --cert "$TMP_CERT" \
  --scope namespace-wide \
  --format yaml < "$TMP_SECRET" > "$TMP_SEALED"

echo "Applying SealedSecret to cluster..."
kubectl apply -f "$TMP_SEALED" -n "$NAMESPACE"

echo "Updating $VALUES_FILE..."
# Utilisation de Python pour mettre à jour proprement le YAML (sans détruire le fichier)
python3 - "$TMP_SEALED" "$VALUES_FILE" << 'EOF'
import yaml
import sys

sealed_file, values_file = sys.argv[1], sys.argv[2]

# 1. Lire le SealedSecret généré
with open(sealed_file, 'r') as f:
    docs = list(yaml.safe_load_all(f))
    sealed_doc = next(d for d in docs if d and 'spec' in d)
    encrypted_data = sealed_doc['spec']['encryptedData']

# 2. Lire le values.yaml actuel
with open(values_file, 'r') as f:
    values = yaml.safe_load(f)

# 3. S'assurer que le chemin env.private existe
if 'env' not in values:
    values['env'] = {}
if 'private' not in values['env']:
    values['env']['private'] = {}

# 4. Mettre à jour / ajouter les clés chiffrées
for key, enc_val in encrypted_data.items():
    values['env']['private'][key] = enc_val

# 5. Réécrire le fichier values.yaml
with open(values_file, 'w') as f:
    yaml.dump(values, f, sort_keys=False, default_flow_style=False)

print(f"✅ Succès : {len(encrypted_data)} clés mises à jour dans {values_file}")
EOF

echo "Done."