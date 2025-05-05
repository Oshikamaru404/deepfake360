# Utilise une image Python officielle
FROM python:3.9-slim

# Mettre à jour les packages et installer CMake
RUN apt-get update && apt-get install -y cmake

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application
COPY . .

# Exposer le port sur lequel l'application va tourner (ajuste en fonction de ton app)
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["python", "app.py"]
