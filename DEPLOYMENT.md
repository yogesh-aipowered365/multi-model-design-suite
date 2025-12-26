# Streamlit Cloud Deployment Guide

## Quick Start: Deploy in 5 Minutes

### Step 1: Prepare Your GitHub Repository ✅
Your code is already pushed to: `https://github.com/yogesh-aipowered365/multi-model-design-suite.git`

### Step 2: Sign Up for Streamlit Cloud
1. Go to **[streamlit.io/cloud](https://streamlit.io/cloud)**
2. Click **"Sign in"** (top right)
3. Click **"Continue with GitHub"**
4. Authorize Streamlit to access your GitHub repositories
5. You'll be redirected to the Streamlit Cloud dashboard

### Step 3: Deploy Your App
1. Click **"New app"** (top left button)
2. Select your repository: `multi-model-design-suite`
3. Select the branch: `main`
4. Select the main file: `app.py`
5. Click **"Deploy"**

Streamlit will automatically:
- Install dependencies from `requirements.txt`
- Run your app on a shared URL (e.g., `https://your-app-name.streamlit.app`)
- Make it publicly accessible

### Step 4: Configure Secrets (Important for API Keys)
1. After deployment, go to your **App Settings** (gear icon)
2. Click **"Secrets"**
3. Add your environment variables in TOML format:
   ```toml
   OPENROUTER_API_KEY = "your-actual-api-key"
   VISION_MODEL = "openai/gpt-4o"
   ```
4. Click **"Save"**

**Note**: Users can also provide their OpenRouter API key directly in the app sidebar (BYOK model), so you don't *need* to set this in Secrets.

### Step 5: Share Your App
Your app URL will be: `https://multi-model-design-suite.streamlit.app` (or similar)

Share this link with users!

---

## Deployment Architecture

```
GitHub Repository (code)
    ↓
Streamlit Cloud (hosting)
    ├─ Runs: Python 3.9+
    ├─ Installs: requirements.txt dependencies
    ├─ Loads: .streamlit/config.toml (UI settings)
    ├─ Loads: Secrets (API keys)
    └─ Serves: Your app at https://your-app.streamlit.app
```

---

## Environment Variables & Secrets

### Option 1: Streamlit Cloud Secrets (Recommended)
1. Go to App Settings > Secrets
2. Add `OPENROUTER_API_KEY = "sk-or-v1-..."`
3. In your app, access via: `st.secrets["OPENROUTER_API_KEY"]`

### Option 2: User Input (BYOK - Already Implemented)
- Users enter API key directly in the app sidebar
- No hardcoded credentials needed

### Option 3: .env File (NOT RECOMMENDED for Cloud)
- Won't work on Streamlit Cloud (no .env file in deployed environment)
- Use Secrets instead

---

## Performance & Limits

### Streamlit Cloud Free Tier
- **CPU**: Shared container
- **Memory**: 1 GB RAM
- **Storage**: 1 GB
- **Concurrent Users**: Up to 5 simultaneously
- **Redeployment**: Automatic on git push
- **Uptime**: 24/7 (with auto-sleep after 7 days of inactivity)

### Expected Performance
- **Cold Start**: ~30-60 seconds first load
- **Warm Start**: ~5-10 seconds after caching
- **Model Loading**: FAISS index loads on first request (~5 sec)
- **Analysis Time**: 10-30 seconds per design (depends on OpenRouter latency)

### If You Need More:
Upgrade to Streamlit Pro for:
- Dedicated resources
- Custom domain
- Private apps
- Priority support

---

## Troubleshooting

### App Won't Deploy
**Issue**: Deployment fails with dependency error
**Solution**:
```bash
# Ensure requirements.txt is up-to-date:
pip freeze > requirements.txt

# Commit and push:
git add requirements.txt
git commit -m "Update requirements"
git push origin main

# Then reauthorize Streamlit to see the latest code
```

### "ModuleNotFoundError"
**Issue**: Missing imports when deployed
**Solution**:
1. Check that all imports are in `requirements.txt`
2. Rebuild the Docker image (click **Rerun** or **Restart**)
3. View logs: Click **Manage App** > **View Logs**

### App Runs Locally But Not on Cloud
**Issue**: Works in development but fails on Streamlit Cloud
**Solution**:
1. Check `.env` dependency—move to Secrets instead
2. Verify absolute paths (use relative paths for assets)
3. View logs in Streamlit Cloud dashboard

### Out of Memory
**Issue**: App crashes with memory error
**Solution**:
- Reduce FAISS index size
- Implement caching for embeddings
- Upgrade to Streamlit Pro

### Slow Performance
**Issue**: App responds slowly
**Solution**:
1. Check OpenRouter API latency
2. Reduce RAG `top_k` slider (fewer patterns = faster)
3. Clear Streamlit cache: Settings > Clear cache
4. Upgrade to Pro for dedicated resources

---

## Advanced Configuration

### Custom Domain
1. Upgrade to **Streamlit Pro**
2. Go to App Settings > Custom Domain
3. Point your domain CNAME to Streamlit

### Authentication (Optional)
Add authentication to restrict access:

```python
import streamlit as st
from streamlit_authenticator import Authenticate

authenticator = Authenticate(...)
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    # Show app content
    st.write("Welcome!")
else:
    st.error("Invalid credentials")
```

### Monitoring & Analytics
1. Go to **Streamlit Analytics** (dashboard home)
2. View user sessions, page views, and errors
3. Monitor resource usage

---

## Post-Deployment Checklist

- ✅ App deploys without errors
- ✅ App loads in under 60 seconds
- ✅ API key input works in sidebar
- ✅ Analysis runs successfully
- ✅ No sensitive data in logs
- ✅ Logo displays correctly
- ✅ All tabs render properly
- ✅ Results download works

---

## Update & Maintenance

### Deploy Updates
1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Feature: Add new capability"
   git push origin main
   ```
3. Streamlit automatically redeploys within 1-2 minutes

### Monitor App Health
1. Check **Manage App** > **View Logs** regularly
2. Monitor **Streamlit Analytics** for errors
3. Set up email alerts for crashes

### Rollback Changes
```bash
# Revert to previous commit
git revert HEAD
git push origin main

# Streamlit will automatically re-deploy the previous version
```

---

## Security Best Practices

1. **Never commit `.env`** (already in `.gitignore`)
2. **Use Secrets** for API keys (not in code)
3. **Enable** HTTPS (Streamlit Cloud default)
4. **Validate user input** (sanitize image uploads)
5. **Monitor logs** for suspicious activity
6. **Rate limit** API calls to prevent abuse
7. **Use BYOK model** so users control their own API keys

---

## Support & Resources

- **Streamlit Cloud Docs**: [docs.streamlit.io/streamlit-cloud](https://docs.streamlit.io/streamlit-cloud)
- **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: [github.com/yogesh-aipowered365/multi-model-design-suite/issues](https://github.com/yogesh-aipowered365/multi-model-design-suite)

---

## Example Deployed App

Once deployed, your app will be available at:
```
https://multi-model-design-suite.streamlit.app
```

Or with a custom domain (Pro):
```
https://design-analyzer.yourcompany.com
```

Users can:
1. Visit the link
2. Enter their OpenRouter API key
3. Upload design images
4. Get AI analysis instantly

---

**Made with ❤️ for designers and product teams**
