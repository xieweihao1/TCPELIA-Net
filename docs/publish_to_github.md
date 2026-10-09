# How to publish TCPELIA on GitHub (Windows / PowerShell)

## 1. Check the release checklist

First read [`release_checklist.md`](release_checklist.md), especially copyright ownership, coauthor/institutional IP requirements, and the proposed MIT license. GitHub visibility **Public** and an open-source license are different choices: check both before publishing.

## 2. Create the repository

1. Sign in at https://github.com/ and click **New repository**.
2. Repository name: `TCPELIA`.
3. Description: `A feedback-driven PyTorch attention module with predictive-error computation, lateral inhibition, and dynamic resource gating.`
4. Select **Public** if you have authorization to release the code.
5. Do **not** create a second README, `.gitignore`, or LICENSE during creation, because these already exist locally.

## 3. Publish the contents

Unzip `TCPELIA_GitHub.zip`, open PowerShell in the extracted `TCPELIA_GitHub` folder, and run:

```powershell
git init
git add .
git commit -m "Initial public release of TCPELIA attention"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/TCPELIA.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username. GitHub may ask you to authenticate using your credential manager or another supported method. Never paste access tokens into code files or commit history.

If you have already created a local git repository, do not reinitialize or overwrite it indiscriminately; check `git status` and `git remote -v` first.

## 4. Validate the public release

```powershell
python -m pip install -e .
python -m pytest -q
python -m examples.quickstart
```

On GitHub, confirm that the README renders, the Mermaid architecture diagram is visible, and the workflow under **Actions** completes. The repository's standalone PyTorch test does not by itself prove compatibility with a modified YOLO26 installation.

## 5. Add research-specific artifacts later

When ready, add confirmed figures, exact experiment configurations, fully reproducible training/evaluation scripts, data access instructions, and the final paper citation. Avoid publishing unsupported accuracy claims, absolute Windows machine paths, or private training material.
