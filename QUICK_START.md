# 🚀 Quick Deploy Guide (NO DOCKER!)

## ✅ Your App is Ready - Just 2 Simple Steps!

---

## 🎯 STEP 1: Deploy Frontend to Vercel (5 min)

### Go to: https://vercel.com

1. Click **"Sign Up"** or **"Log In"** with GitHub
2. Click **"Add New..."** → **"Project"**
3. Find and select: `krithika-029/MAJOR-PROJECT-FINAL`
4. Click **"Import"**

### Configure Settings:
- **Framework**: Vite
- **Root Directory**: `frontend-react`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

### Click "Deploy" button

✅ **Your frontend is live in 2-3 minutes!**

**Copy your Vercel URL** (e.g., `https://ki67-project.vercel.app`)

---

## 🔧 STEP 2: Deploy Backend to Render (15 min)

### Go to: https://render.com

1. Click **"Get Started"** → Sign in with GitHub
2. Click **"New +"** → **"Web Service"**
3. Find and select: `krithika-029/MAJOR-PROJECT-FINAL`
4. Click **"Connect"**

### Configure Settings:
- **Name**: `ki67-backend`
- **Environment**: `Docker`
- **Branch**: `main`
- **Instance Type**: `Free`

### Add Environment Variable:
Click **"Advanced"** → **"Add Environment Variable"**
```
Name: ALLOWED_ORIGINS
Value: https://ki67-project.vercel.app
```
(Use YOUR Vercel URL from Step 1!)

### Click "Create Web Service"

⏳ **Wait 15-20 minutes** (first deploy is slow, pulling AI model)

✅ **Your backend is live!**

**Copy your Render URL** (e.g., `https://ki67-backend.onrender.com`)

---

## 🔗 STEP 3: Connect Them (2 min)

### Go back to Vercel:

1. Open your project dashboard
2. Click **"Settings"** → **"Environment Variables"**
3. Click **"Add New"**
   ```
   Name: VITE_API_URL
   Value: https://ki67-backend.onrender.com
   ```
   (Use YOUR Render URL from Step 2!)

4. Go to **"Deployments"** tab
5. Click **"..."** menu on latest → **"Redeploy"**
6. Click **"Redeploy"** to confirm

⏳ **Wait 2 minutes**

---

## 🎉 DONE! Your App is Live!

Visit your Vercel URL: `https://ki67-project.vercel.app`

✅ Upload medical images
✅ Get AI analysis
✅ Download reports
✅ View history

**Everything works - NO DOCKER NEEDED!** 🎊

---

## 💡 What You Just Did

```
Frontend (Vercel)          Backend (Render)
┌─────────────────┐       ┌─────────────────┐
│   React App     │ ────▶ │  Flask API      │
│   (Fast CDN)    │       │  AI Model       │
│   Static Files  │ ◀──── │  Database       │
└─────────────────┘       └─────────────────┘
      FREE                       FREE
```

---

## ⚠️ Important Notes

### First Use:
- **Frontend**: Instant! Always fast ⚡
- **Backend**: First request takes 30-60 seconds (it's sleeping)
- After wake-up: Fast! ⚡

### After 15 Minutes of Inactivity:
- Backend sleeps (free tier)
- Next request wakes it up (30-60 sec wait)
- Frontend stays fast always!

**Tip**: Keep a tab open to keep backend awake!

---

## 🔄 Update Your App Later

Just push to GitHub:
```powershell
git add .
git commit -m "Update"
git push origin main
```

**Both Vercel AND Render auto-deploy!** No manual work needed! 🚀

---

## 🆘 Troubleshooting

### "Cannot connect to backend"
- Check VITE_API_URL in Vercel matches your Render URL
- Check ALLOWED_ORIGINS in Render matches your Vercel URL
- Redeploy frontend after changing env vars

### "Backend is slow"
- It's sleeping! First request wakes it (30-60 sec)
- Then it's fast for 15 minutes

### "Model loading error"
- Render is still deploying (wait full 20 minutes)
- Check Render logs for errors

---

## 📊 Cost Breakdown

| Service | Cost | What You Get |
|---------|------|--------------|
| **Vercel** | $0 | 100 GB bandwidth/month |
| **Render** | $0 | 750 hours/month |
| **Total** | **$0** | Full AI medical app! |

---

## ✨ What's Included FREE

✅ Automatic HTTPS (secure)
✅ Custom domain support
✅ Auto-deploy from GitHub
✅ Unlimited frontend traffic
✅ Backend with AI inference
✅ File uploads & database
✅ PDF report generation

---

## 🎯 Start Now!

1. Open Vercel: https://vercel.com
2. Deploy frontend (5 min)
3. Open Render: https://render.com
4. Deploy backend (15 min)
5. Connect them (2 min)
6. **Use your live app!** 🎉

**No Docker. No complicated setup. Just works!** ✨
