# 🚂 Deploy to Railway - Step by Step Guide (5 Minutes!)

## ✅ Prerequisites (Already Done!)
- ✅ GitHub repository: `krithika-029/MAJOR-PROJECT-FINAL`
- ✅ Dockerfile fixed and pushed
- ✅ Code committed to main branch

## 🚀 Deployment Steps

### Step 1: Sign Up for Railway (1 minute)

1. Open your web browser and go to: **https://railway.app**

2. Click the **"Login"** button (top right)

3. Click **"Login with GitHub"**

4. Enter your GitHub credentials and click **"Authorize Railway"**

### Step 2: Create New Project (2 minutes)

1. You'll see the Railway dashboard

2. Click the big **"New Project"** button

3. Select **"Deploy from GitHub repo"**

4. You'll see a list of your repositories

5. Find and click: **`krithika-029/MAJOR-PROJECT-FINAL`**

6. Railway will automatically:
   - ✅ Detect your Dockerfile
   - ✅ Start building the Docker image
   - ✅ Deploy the container

### Step 3: Configure Your Service (1 minute)

1. Wait for the build to start (you'll see logs appearing)

2. Click on your service card (it will say "ki67-malignancy-classification" or similar)

3. Go to the **"Settings"** tab

4. Scroll down to **"Networking"** section

5. Click **"Generate Domain"** button

6. Railway will give you a public URL like:
   ```
   https://ki67-malignancy-classification-production.up.railway.app
   ```

7. **Copy this URL!** This is your live app address 🎉

### Step 4: Wait for Build to Complete (5-10 minutes)

1. Go back to the **"Deployments"** tab

2. Click on the active deployment (green dot)

3. Watch the build logs. You'll see:
   ```
   Building frontend...
   Installing Python dependencies...
   Copying model checkpoint...
   Build successful ✓
   ```

4. Once you see **"✅ Deployed"** status, your app is live!

### Step 5: Test Your App (1 minute)

1. Click the domain URL you generated in Step 3

2. You should see your Ki-67 Medical Diagnostic System!

3. Test by uploading an image

4. Check that analysis works

## 🎉 Congratulations!

Your app is now live on the internet! Anyone with the URL can access it.

**Your Railway URL**: `https://your-app-name.up.railway.app`

---

## 💡 Pro Tips

### Check Your Free Credit
- Click your profile icon (top right)
- Go to "Usage"
- Monitor your $5 free monthly credit

### View Logs
- Click on your service
- Go to "Deployments" tab
- Click on a deployment to see logs
- Useful for debugging issues

### Update Your App
Whenever you push code to GitHub:
```powershell
git add .
git commit -m "Update feature"
git push origin main
```
Railway will **automatically rebuild and redeploy**! 🚀

### Add Environment Variables (Optional)
- Go to "Variables" tab
- Add variables like:
  - `PORT=5001`
  - `FLASK_ENV=production`

### Pause Your App (Save Credits)
- Go to "Settings" tab
- Scroll to bottom
- Click "Delete Service" when not using
- Redeploy anytime from your GitHub repo

---

## 🆘 Troubleshooting

### Build Failed?
**Check:**
1. Dockerfile is in repository root ✅ (it is!)
2. Model checkpoint was pulled with Git LFS
   ```powershell
   git lfs pull
   git push origin main
   ```

### App Not Loading?
**Try:**
1. Wait 2-3 minutes after deployment completes
2. Check deployment logs for errors
3. Visit `/api/health` endpoint first
4. Hard refresh browser (Ctrl + Shift + R)

### Out of Credits?
**Options:**
1. Wait until next month (resets monthly)
2. Delete unused deployments to save credits
3. Use another free service (Render, Fly.io)

---

## 📊 What Railway Provides FREE

- ✅ $5 credit per month (~500 hours)
- ✅ Automatic HTTPS
- ✅ Custom domains
- ✅ Automatic deployments from GitHub
- ✅ Persistent storage (volumes)
- ✅ Environment variables
- ✅ Deployment logs

---

## 🔗 Quick Links

- **Railway Dashboard**: https://railway.app/dashboard
- **Your GitHub Repo**: https://github.com/krithika-029/MAJOR-PROJECT-FINAL
- **Railway Docs**: https://docs.railway.app

---

## ⏱️ Total Time: 5 Minutes!

1. Sign up with GitHub (1 min) ✅
2. Deploy from repo (2 min) ✅
3. Generate domain (1 min) ✅
4. Wait for build (5-10 min background) ⏳
5. Access live app! 🎉

**No credit card required!** 💳❌

---

Need help? The deployment logs will show you exactly what's happening at each step!
