# Deploy Ki-67 App (Split Architecture - NO DOCKER NEEDED!)

## 🎯 Simple 2-Step Deployment (Both FREE!)

Your app needs both frontend (React) and backend (Python AI). We'll deploy them separately:

- **Frontend** → Vercel (free, instant)
- **Backend** → Render (free, handles Python/AI)

---

## 📱 Step 1: Deploy Frontend to Vercel (5 minutes)

### Instructions:

1. **Go to Vercel**: https://vercel.com

2. **Sign in with GitHub**

3. **Click "Add New..."** → **"Project"**

4. **Import** your repository: `krithika-029/MAJOR-PROJECT-FINAL`

5. **Configure**:
   ```
   Framework Preset: Vite
   Root Directory: frontend-react
   Build Command: npm run build
   Output Directory: dist
   ```

6. **Add Environment Variable**:
   ```
   Name: VITE_API_URL
   Value: https://ki67-backend.onrender.com
   ```
   *(We'll update this after deploying backend)*

7. **Click "Deploy"**

8. **Wait 2-3 minutes** → Your frontend is live! 🎉

9. **Copy the URL** (e.g., `https://ki67-project.vercel.app`)

---

## 🔧 Step 2: Deploy Backend to Render (15 minutes)

### Instructions:

1. **Go to Render**: https://render.com

2. **Sign in with GitHub**

3. **Click "New +"** → **"Web Service"**

4. **Connect repository**: `krithika-029/MAJOR-PROJECT-FINAL`

5. **Configure**:
   ```
   Name: ki67-backend
   Environment: Docker
   Region: Choose closest to you
   Branch: main
   Instance Type: Free
   ```

6. **Add Environment Variables**:
   ```
   FLASK_ENV=production
   PORT=5001
   ALLOWED_ORIGINS=https://ki67-project.vercel.app
   ```
   *(Replace with your Vercel URL from Step 1)*

7. **Click "Create Web Service"**

8. **Wait 15-20 minutes** (large model file)

9. **Copy the backend URL** (e.g., `https://ki67-backend.onrender.com`)

---

## 🔗 Step 3: Connect Frontend to Backend (2 minutes)

1. **Go back to Vercel dashboard**

2. **Click your project** → **"Settings"** → **"Environment Variables"**

3. **Update** `VITE_API_URL`:
   ```
   Value: https://ki67-backend.onrender.com
   ```
   *(Use your actual Render URL)*

4. **Go to "Deployments"** tab

5. **Click "..." menu** on latest deployment → **"Redeploy"**

6. **Wait 1-2 minutes** for rebuild

---

## ✅ Done! Your App is Live!

- **Frontend URL**: `https://ki67-project.vercel.app`
- **Backend URL**: `https://ki67-backend.onrender.com`
- **Full app works** through the frontend URL!

---

## 🎉 Benefits of This Approach

✅ **No Docker needed** - both platforms handle everything
✅ **Completely FREE** - no credit card required
✅ **Auto-deploy** - push to GitHub = automatic updates
✅ **Fast frontend** - Vercel CDN is super fast
✅ **Handles AI model** - Render supports Python/PyTorch
✅ **Git LFS works** - Render handles your model file

---

## ⚙️ How It Works

```
User visits Vercel URL
     ↓
Loads React Frontend (fast!)
     ↓
User uploads image
     ↓
Frontend sends to Render Backend
     ↓
AI model processes (on Render)
     ↓
Results sent back to Frontend
     ↓
User sees analysis!
```

---

## 📊 Platform Details

### Vercel (Frontend)
- **Free Tier**: 100 GB bandwidth/month
- **Speed**: Lightning fast (CDN)
- **Deploys**: Instant (1-2 minutes)
- **Perfect for**: React, Vue, Next.js

### Render (Backend)
- **Free Tier**: 750 hours/month
- **Speed**: Good (spins down after 15 min idle)
- **Deploys**: 15-20 minutes (first time)
- **Perfect for**: Python, Docker, AI models

---

## ⚠️ Free Tier Limitations

### Vercel:
- ✅ Always on
- ✅ No cold starts
- ✅ Unlimited deploys

### Render:
- ⚠️ **Sleeps after 15 min** of inactivity
- ⚠️ **First request takes 30-60 seconds** to wake up
- ✅ Stays awake while in use
- ✅ 750 hours/month (plenty for testing)

**Tip**: The frontend loads instantly, only backend has cold starts!

---

## 🔄 Updating Your App

Just push to GitHub:

```powershell
cd "c:\Users\Hp\Downloads\Ki67-Malignancy-Assessment-System\Ki67-Malignancy-Classification-Code-Only (1)"
git add .
git commit -m "Update feature"
git push origin main
```

**Both Vercel and Render auto-deploy!** 🚀

---

## 🆘 Troubleshooting

### Frontend deploys but can't connect to backend?

**Fix**: Update `VITE_API_URL` in Vercel environment variables with correct Render URL

### Backend shows "Model loading error"?

**Fix**: Render is pulling Git LFS automatically. Wait 20 minutes for first deploy.

### Backend is slow?

**Fix**: It's sleeping! First request wakes it up (30-60 sec), then it's fast.

---

## 💡 Alternative: All-in-One Render

If you want everything in one place:

1. Deploy to Render as **Web Service** (Docker)
2. Skip Vercel entirely
3. Render serves both frontend + backend
4. Simpler, but backend serves static files (slightly slower frontend)

---

## 🎯 Which Approach?

| Feature | Split (Vercel + Render) | All-in-One (Render only) |
|---------|------------------------|--------------------------|
| **Setup** | 2 platforms | 1 platform |
| **Frontend Speed** | ⚡ Blazing fast | 🐢 Good |
| **Backend** | ✅ Same | ✅ Same |
| **Free Tier** | ✅ Both free | ✅ Free |
| **Cold Starts** | Frontend always on | Everything sleeps |
| **Best For** | Production | Simple setup |

**My Recommendation**: **Split architecture** (Vercel + Render) for best performance!

---

## 📋 Quick Start Checklist

- [ ] Deploy frontend to Vercel
- [ ] Deploy backend to Render
- [ ] Update VITE_API_URL in Vercel
- [ ] Redeploy frontend on Vercel
- [ ] Test the app!

**Total Time**: ~25 minutes
**Cost**: $0 (completely free!)
**Docker**: Not needed! 🎉

---

Ready to start? Follow Step 1 first (Vercel frontend) - it takes only 5 minutes!
