# Étape 1: Utilisation d'une image Python de base
FROM python:3.12-slim

# Étape 2: Installation des dépendances système
# On met à jour les paquets et on installe cmake et autres dépendances nécessaires
RUN apt-get update && \
    apt-get install -y cmake build-essential libboost-all-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Étape 3: Définition du répertoire de travail
WORKDIR /app

# Étape 4: Copier tous les fichiers du projet dans le conteneur
COPY . /app

# Étape 5: Installation de pip et des dépendances Python depuis requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Étape 6: Définition de la commande à exécuter lorsque le conteneur démarre
CMD ["python", "main.py"]
