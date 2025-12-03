# Ki-67 Medical Diagnostic System - Deployment Guide

## Table of Contents
- [Docker Deployment](#docker-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
  - [AWS](#aws-deployment)
  - [Azure](#azure-deployment)
  - [Google Cloud Platform](#google-cloud-platform)
  - [Heroku](#heroku-deployment)
  - [Railway](#railway-deployment)
  - [DigitalOcean](#digitalocean-deployment)
- [Production Optimizations](#production-optimizations)
- [Environment Variables](#environment-variables)

---

## Docker Deployment

### Prerequisites
- Docker installed (version 20.10+)
- Docker Compose installed (version 2.0+)
- Git LFS configured for model checkpoint

### Quick Start (Local Testing)

1. **Clone the repository**
   ```bash
   git clone https://github.com/krithika-029/MAJOR-PROJECT-FINAL.git
   cd MAJOR-PROJECT-FINAL
   git lfs pull  # Download model checkpoint
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose build
   docker-compose up
   ```

3. **Access the application**
   - Web UI: http://localhost:5001
   - API Health: http://localhost:5001/api/health

### Production Deployment with Docker

1. **Build the Docker image**
   ```bash
   docker build -t ki67-diagnostic:latest .
   ```

2. **Run the container with persistent volumes**
   ```bash
   docker run -d \
     --name ki67-app \
     -p 5001:5001 \
     -v $(pwd)/ki67.db:/app/ki67.db \
     -v $(pwd)/uploads:/app/uploads \
     -v $(pwd)/results:/app/results \
     --restart unless-stopped \
     ki67-diagnostic:latest
   ```

3. **With nginx reverse proxy (recommended for production)**
   ```bash
   docker-compose --profile production up -d
   ```

### Docker Commands

- **View logs**: `docker-compose logs -f ki67-app`
- **Stop services**: `docker-compose down`
- **Rebuild after changes**: `docker-compose up -d --build`
- **Shell access**: `docker exec -it ki67-app bash`

---

## Cloud Platform Deployment

### AWS Deployment

#### Option 1: Amazon ECS (Elastic Container Service)

1. **Install AWS CLI and configure credentials**
   ```bash
   aws configure
   ```

2. **Create ECR repository**
   ```bash
   aws ecr create-repository --repository-name ki67-diagnostic
   ```

3. **Build and push Docker image**
   ```bash
   # Get login credentials
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and tag
   docker build -t ki67-diagnostic .
   docker tag ki67-diagnostic:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ki67-diagnostic:latest
   
   # Push to ECR
   docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ki67-diagnostic:latest
   ```

4. **Deploy to ECS Fargate**
   - Create ECS cluster in AWS Console
   - Create Task Definition:
     - Container: Use ECR image URI
     - Memory: 4GB (minimum for AI model)
     - CPU: 2 vCPU
     - Port mappings: 5001
   - Create Service with Load Balancer
   - Configure EFS volume for persistent storage

#### Option 2: AWS Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize and deploy**
   ```bash
   eb init -p docker ki67-diagnostic
   eb create ki67-prod-env
   eb open
   ```

3. **Configure environment**
   - Increase instance size to t3.medium or larger
   - Add EBS volume for persistent storage
   - Configure load balancer health check: `/api/health`

#### Option 3: AWS EC2 (Manual Setup)

1. **Launch EC2 instance**
   - AMI: Amazon Linux 2 or Ubuntu 22.04
   - Instance type: t3.large (minimum 2 vCPU, 8GB RAM)
   - Storage: 30GB+ EBS volume
   - Security Group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS)

2. **Connect and install Docker**
   ```bash
   ssh -i your-key.pem ec2-user@your-instance-ip
   
   # Install Docker
   sudo yum update -y
   sudo yum install docker git git-lfs -y
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **Deploy the application**
   ```bash
   git clone https://github.com/krithika-029/MAJOR-PROJECT-FINAL.git
   cd MAJOR-PROJECT-FINAL
   git lfs pull
   docker-compose up -d
   ```

4. **Configure domain (optional)**
   - Point your domain to EC2 public IP
   - Use Let's Encrypt for SSL: `certbot --nginx -d your-domain.com`

**Cost Estimate**: $50-150/month depending on instance size and storage

---

### Azure Deployment

#### Option 1: Azure Container Instances (Fastest)

1. **Install Azure CLI**
   ```bash
   az login
   ```

2. **Create resource group**
   ```bash
   az group create --name ki67-rg --location eastus
   ```

3. **Create Azure Container Registry**
   ```bash
   az acr create --resource-group ki67-rg --name ki67acr --sku Basic
   az acr login --name ki67acr
   ```

4. **Build and push image**
   ```bash
   docker build -t ki67acr.azurecr.io/ki67-diagnostic:latest .
   docker push ki67acr.azurecr.io/ki67-diagnostic:latest
   ```

5. **Deploy to ACI**
   ```bash
   az container create \
     --resource-group ki67-rg \
     --name ki67-app \
     --image ki67acr.azurecr.io/ki67-diagnostic:latest \
     --cpu 2 --memory 4 \
     --ports 5001 \
     --dns-name-label ki67-diagnostic \
     --registry-username ki67acr \
     --registry-password $(az acr credential show --name ki67acr --query "passwords[0].value" -o tsv)
   ```

6. **Access application**
   ```
   http://ki67-diagnostic.eastus.azurecontainer.io:5001
   ```

#### Option 2: Azure App Service

1. **Create App Service Plan**
   ```bash
   az appservice plan create \
     --name ki67-plan \
     --resource-group ki67-rg \
     --sku B2 \
     --is-linux
   ```

2. **Deploy container**
   ```bash
   az webapp create \
     --resource-group ki67-rg \
     --plan ki67-plan \
     --name ki67-webapp \
     --deployment-container-image-name ki67acr.azurecr.io/ki67-diagnostic:latest
   
   az webapp config appsettings set \
     --resource-group ki67-rg \
     --name ki67-webapp \
     --settings WEBSITES_PORT=5001
   ```

**Cost Estimate**: $50-200/month depending on service tier

---

### Google Cloud Platform

#### Option 1: Cloud Run (Serverless, Recommended)

1. **Install gcloud CLI and authenticate**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Build and deploy with one command**
   ```bash
   gcloud run deploy ki67-diagnostic \
     --source . \
     --platform managed \
     --region us-central1 \
     --memory 4Gi \
     --cpu 2 \
     --port 5001 \
     --allow-unauthenticated
   ```

3. **Access your application**
   - Cloud Run will provide a URL like: `https://ki67-diagnostic-xxxxx-uc.a.run.app`

**Pros**: Auto-scaling, pay-per-use, HTTPS included  
**Cost Estimate**: $20-100/month (only pay when processing requests)

#### Option 2: Google Kubernetes Engine (GKE)

1. **Create GKE cluster**
   ```bash
   gcloud container clusters create ki67-cluster \
     --num-nodes=2 \
     --machine-type=n1-standard-2 \
     --zone us-central1-a
   ```

2. **Build and push to Container Registry**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ki67-diagnostic
   ```

3. **Deploy to GKE**
   ```bash
   kubectl create deployment ki67-app --image=gcr.io/YOUR_PROJECT_ID/ki67-diagnostic
   kubectl expose deployment ki67-app --type=LoadBalancer --port=80 --target-port=5001
   ```

**Cost Estimate**: $100-300/month (cluster + load balancer)

---

### Heroku Deployment

1. **Install Heroku CLI**
   ```bash
   heroku login
   ```

2. **Create Heroku app**
   ```bash
   heroku create ki67-diagnostic
   ```

3. **Set stack to container**
   ```bash
   heroku stack:set container -a ki67-diagnostic
   ```

4. **Create `heroku.yml` in project root**
   ```yaml
   build:
     docker:
       web: Dockerfile
   run:
     web: python backend/app.py
   ```

5. **Deploy**
   ```bash
   git add heroku.yml
   git commit -m "Add Heroku container config"
   git push heroku main
   ```

6. **Scale dyno (minimum Performance-M for AI model)**
   ```bash
   heroku ps:scale web=1:performance-m
   ```

7. **View logs**
   ```bash
   heroku logs --tail
   ```

**Limitations**: 
- Large model checkpoint (147MB) requires Git LFS
- Performance-M dyno costs $250/month (required for 2.5GB RAM)

**Cost Estimate**: $250+/month

---

### Railway Deployment

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Initialize project**
   ```bash
   railway init
   ```

3. **Deploy**
   ```bash
   railway up
   ```

4. **Configure custom domain (optional)**
   ```bash
   railway domain
   ```

**Pros**: 
- Simple deployment
- $5 free credit monthly
- Automatic HTTPS
- Git-based deployments

**Cost Estimate**: $10-50/month (pay-as-you-go after free tier)

---

### DigitalOcean Deployment

#### Option 1: App Platform (PaaS)

1. **Connect GitHub repository**
   - Go to DigitalOcean App Platform console
   - Select "Create App from GitHub"
   - Authorize and select `MAJOR-PROJECT-FINAL` repository

2. **Configure app**
   - Type: Docker Web Service
   - Dockerfile path: `Dockerfile`
   - HTTP port: 5001
   - Instance size: Professional (2GB RAM minimum)

3. **Add persistent volume**
   - Mount path: `/app/uploads` and `/app/results`
   - Size: 10GB+

**Cost Estimate**: $24-48/month

#### Option 2: Droplet (VPS)

1. **Create Droplet**
   - Image: Docker on Ubuntu 22.04
   - Plan: Basic - 2 vCPU, 4GB RAM ($24/month)
   - Add SSH key

2. **SSH and deploy**
   ```bash
   ssh root@your-droplet-ip
   git clone https://github.com/krithika-029/MAJOR-PROJECT-FINAL.git
   cd MAJOR-PROJECT-FINAL
   git lfs pull
   docker-compose up -d
   ```

3. **Configure firewall**
   ```bash
   ufw allow 22
   ufw allow 80
   ufw allow 443
   ufw enable
   ```

**Cost Estimate**: $24-48/month

---

## Production Optimizations

### 1. Use Gunicorn (Production WSGI Server)

**Update `backend/app.py`** - Add at the bottom:
```python
if __name__ == "__main__":
    # Production: Use gunicorn
    # gunicorn -w 2 -b 0.0.0.0:5001 --timeout 300 backend.app:app
    
    # Development: Use Flask dev server
    app.run(host="0.0.0.0", port=5001, debug=False)
```

**Update `Dockerfile`** - Change CMD:
```dockerfile
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "--timeout", "300", "backend.app:app"]
```

**Add to `requirements.txt`**:
```
gunicorn>=21.2.0
```

### 2. Nginx Reverse Proxy

**Create `nginx.conf`**:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream ki67_backend {
        server ki67-app:5001;
    }

    server {
        listen 80;
        server_name _;
        client_max_body_size 50M;

        location / {
            proxy_pass http://ki67_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        location /static/ {
            alias /app/frontend-react/dist/assets/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

**Enable nginx in docker-compose**:
```bash
docker-compose --profile production up -d
```

### 3. Environment Variables

**Create `.env` file**:
```env
# Application
FLASK_ENV=production
PORT=5001

# Database
DB_PATH=ki67.db

# Uploads
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=52428800  # 50MB

# Model
MODEL_PATH=models/ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt
DEVICE=cpu  # or cuda if GPU available

# Logging
LOG_LEVEL=INFO
```

**Update `docker-compose.yml`**:
```yaml
services:
  ki67-app:
    env_file:
      - .env
```

### 4. SSL/TLS Certificate (Production)

**Using Let's Encrypt with Certbot**:
```bash
# Install certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (already configured by certbot)
sudo certbot renew --dry-run
```

### 5. Database Backups

**Create backup script `backup.sh`**:
```bash
#!/bin/bash
BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p $BACKUP_DIR

# Backup database
cp ki67.db "$BACKUP_DIR/ki67_$TIMESTAMP.db"

# Backup uploads and results
tar -czf "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" uploads/ results/

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete
```

**Add to crontab (daily at 2 AM)**:
```bash
crontab -e
0 2 * * * /path/to/backup.sh
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Application port | `5001` |
| `FLASK_ENV` | Flask environment | `production` |
| `DB_PATH` | SQLite database path | `ki67.db` |
| `UPLOAD_FOLDER` | Upload directory | `uploads` |
| `MODEL_PATH` | Model checkpoint path | `models/ki67-point-epoch=68-val_peak_f1_avg=0.8503.ckpt` |
| `DEVICE` | Inference device | `cpu` |
| `MAX_CONTENT_LENGTH` | Max upload size (bytes) | `52428800` (50MB) |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## Monitoring and Health Checks

### Health Check Endpoint
```bash
curl http://localhost:5001/api/health
```

### Docker Health Status
```bash
docker ps --filter "name=ki67-app" --format "{{.Status}}"
```

### View Application Logs
```bash
# Docker Compose
docker-compose logs -f ki67-app

# Docker
docker logs -f ki67-app

# Last 100 lines
docker logs --tail 100 ki67-app
```

---

## Troubleshooting

### Issue: Container runs out of memory
**Solution**: Increase memory allocation in Docker settings or instance size
```yaml
services:
  ki67-app:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Issue: Model checkpoint not found
**Solution**: Ensure Git LFS is configured
```bash
git lfs install
git lfs pull
```

### Issue: Permission denied on volumes
**Solution**: Fix permissions
```bash
chmod -R 755 uploads/ results/
chown -R 1000:1000 uploads/ results/  # Match container user
```

---

## Support and Resources

- **GitHub Repository**: https://github.com/krithika-029/MAJOR-PROJECT-FINAL
- **Docker Documentation**: https://docs.docker.com
- **Flask Production Guide**: https://flask.palletsprojects.com/en/3.0.x/deploying/

---

## Quick Reference

| Platform | Deployment Time | Difficulty | Cost/Month | Best For |
|----------|----------------|------------|------------|----------|
| **Docker Local** | 5 min | Easy | $0 | Development/Testing |
| **Railway** | 10 min | Easy | $10-50 | Quick production deployment |
| **GCP Cloud Run** | 15 min | Easy | $20-100 | Serverless, auto-scaling |
| **DigitalOcean App** | 15 min | Easy | $24-48 | Simple PaaS |
| **AWS ECS** | 30 min | Medium | $50-150 | Scalable containers |
| **Azure ACI** | 20 min | Medium | $50-200 | Quick cloud deployment |
| **Heroku** | 20 min | Medium | $250+ | Managed platform |
| **AWS EC2** | 45 min | Hard | $50-150 | Full control |
| **GKE/AKS** | 60+ min | Hard | $100-300+ | Enterprise, orchestration |
