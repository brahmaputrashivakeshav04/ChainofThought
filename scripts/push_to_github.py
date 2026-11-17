"""Helper script to push ResearchIQ to GitHub using dulwich (pure-Python Git) or system git."""

import sys
import argparse
from pathlib import Path
import dulwich.porcelain as porcelain

REPO_URL = "https://github.com/brahmaputrashivakeshav04/ChainofThought.git"

def main():
    parser = argparse.ArgumentParser(description="Push ResearchIQ repository to GitHub.")
    parser.add_argument("--token", type=str, default="", help="GitHub Personal Access Token (PAT) with repo scope")
    parser.add_argument("--username", type=str, default="brahmaputrashivakeshav04", help="GitHub username")
    args = parser.parse_args()

    repo = porcelain.open_repo(str(Path(__file__).resolve().parent.parent))

    if not args.token:
        print("=" * 65)
        print("ResearchIQ Git Push Helper")
        print("=" * 65)
        print(f"Repository: {REPO_URL}")
        print("Branch:     main\n")
        print("GitHub requires write authentication to push. You have two easy options:\n")
        print("Option 1 (Using a GitHub Personal Access Token):")
        print("  python scripts/push_to_github.py --token <YOUR_GITHUB_PAT>\n")
        print("Option 2 (Using System Git / VS Code):")
        print("  Install Git from https://git-scm.com/download/win (or winget install Git.Git)")
        print("  Then run in terminal: git push -u origin main")
        print("=" * 65)
        return

    auth_url = f"https://{args.username}:{args.token}@github.com/brahmaputrashivakeshav04/ChainofThought.git"
    print(f"[*] Pushing branch 'main' to https://github.com/brahmaputrashivakeshav04/ChainofThought.git...")
    try:
        porcelain.push(repo, auth_url, "refs/heads/main:refs/heads/main")
        print("[+] SUCCESS: Successfully pushed to GitHub!")
        print("    View repository at: https://github.com/brahmaputrashivakeshav04/ChainofThought")
    except Exception as e:
        print(f"[!] Push failed: {type(e).__name__}: {e}")
        print("    Ensure your token has 'repo' write permissions.")

if __name__ == "__main__":
    main()
