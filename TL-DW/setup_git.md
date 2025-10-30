# Git Setup Instructions for TL-DW Project

Follow these steps to push your project to GitHub:

## Step 1: Install Git

Download and install Git from: https://git-scm.com/download/win

## Step 2: Open Terminal/Command Prompt

Navigate to your project directory:
```bash
cd "D:\backkupp\NEWW\Project\TL-DW"
```

## Step 3: Initialize Git Repository

```bash
git init
```

## Step 4: Add Files

```bash
git add app.py helper.py requirements.txt templates/ static/ README.md .gitignore
```

## Step 5: Commit

```bash
git commit -m "Initial commit: TL-DW YouTube Summarizer"
```

## Step 6: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `TOO-LONG-DIDNT-WATCH`
3. Description: "AI-powered YouTube video summarization system with multilingual translation"
4. Set to Public
5. Click "Create repository"

## Step 7: Connect and Push

Replace `AzizKhan2004` with your GitHub username if different:

```bash
git branch -M main
git remote add origin https://github.com/AzizKhan2004/TOO-LONG-DIDNT-WATCH.git
git push -u origin main
```

## Step 8: Alternative (Using SSH)

If you have SSH keys set up:

```bash
git remote add origin git@github.com:AzizKhan2004/TOO-LONG-DIDNT-WATCH.git
git push -u origin main
```

---

## Quick All-in-One Commands (After Git is Installed)

```bash
cd "D:\backkupp\NEWW\Project\TL-DW"
git init
git add app.py helper.py requirements.txt templates/ static/ README.md .gitignore
git commit -m "Initial commit: TL-DW YouTube Summarizer"
git branch -M main
git remote add origin https://github.com/AzizKhan2004/TOO-LONG-DIDNT-WATCH.git
git push -u origin main
```

## Troubleshooting

If you get authentication errors:
1. Use GitHub CLI: `gh auth login`
2. Or use Personal Access Token instead of password
3. Or configure SSH keys

## Need Help?

Visit: https://docs.github.com/en/get-started/getting-started-with-git



