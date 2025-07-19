"""
GitHub Repository Setup Script
==============================
This script helps you set up and push your Healthcare LLM project to GitHub.
"""

import subprocess
import os
import sys

def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        if result.stderr and not check:
            print(f"Warning: {result.stderr.strip()}")
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_git_installed():
    """Check if git is installed."""
    try:
        result = subprocess.run("git --version", shell=True, capture_output=True, text=True)
        print(f"✅ Git is installed: {result.stdout.strip()}")
        return True
    except:
        print("❌ Git is not installed. Please install Git first.")
        return False

def setup_git_repo():
    """Initialize git repository and add files."""
    print("\n🔧 Setting up Git repository...")
    
    # Initialize git repo
    if not os.path.exists(".git"):
        if not run_command("git init", "Initializing Git repository"):
            return False
    else:
        print("✅ Git repository already initialized")
    
    # Add all files
    if not run_command("git add .", "Adding all files to git"):
        return False
    
    # Check git status
    run_command("git status", "Checking git status", check=False)
    
    return True

def create_initial_commit():
    """Create initial commit."""
    print("\n📝 Creating initial commit...")
    
    # Configure git user (if not already configured)
    run_command('git config user.name "Healthcare LLM Developer"', "Setting git username", check=False)
    run_command('git config user.email "developer@healthcare-llm.com"', "Setting git email", check=False)
    
    # Create initial commit
    commit_message = "Initial commit: Healthcare LLM Fine-tuning with LoRA\n\n- Complete LoRA fine-tuning pipeline for healthcare\n- BioGPT-Large model recommendations\n- 2,309 training samples with 866 medical conditions\n- 910 test samples for evaluation\n- Interactive testing and batch processing\n- Comprehensive documentation and analysis"
    
    if not run_command(f'git commit -m "{commit_message}"', "Creating initial commit"):
        return False
    
    return True

def setup_github_remote():
    """Set up GitHub remote repository."""
    print("\n🌐 Setting up GitHub remote...")
    
    print("\n" + "="*60)
    print("📋 GITHUB REPOSITORY SETUP INSTRUCTIONS")
    print("="*60)
    print("1. Go to https://github.com and create a new repository")
    print("2. Repository name: healthcare-llm-finetuning")
    print("3. Description: Healthcare LLM Fine-tuning with LoRA for Medical Diagnosis")
    print("4. Make it Public (recommended) or Private")
    print("5. DO NOT initialize with README (we already have one)")
    print("6. Copy the repository URL")
    
    # Get repository URL from user
    repo_url = input("\n🔗 Enter your GitHub repository URL (e.g., https://github.com/username/healthcare-llm-finetuning.git): ").strip()
    
    if not repo_url:
        print("❌ No repository URL provided")
        return False
    
    # Add remote origin
    if not run_command(f"git remote add origin {repo_url}", "Adding GitHub remote"):
        # Try to set URL if remote already exists
        run_command(f"git remote set-url origin {repo_url}", "Setting GitHub remote URL", check=False)
    
    return True, repo_url

def push_to_github():
    """Push repository to GitHub."""
    print("\n🚀 Pushing to GitHub...")
    
    # Set upstream and push
    if not run_command("git branch -M main", "Setting main branch"):
        return False
    
    if not run_command("git push -u origin main", "Pushing to GitHub"):
        return False
    
    return True

def main():
    """Main setup function."""
    print("🏥 Healthcare LLM GitHub Setup")
    print("=" * 50)
    
    # Check if git is installed
    if not check_git_installed():
        return
    
    # Get current directory info
    current_dir = os.getcwd()
    project_name = os.path.basename(current_dir)
    
    print(f"\n📁 Current project directory: {current_dir}")
    print(f"📋 Project name: {project_name}")
    
    # Check if user wants to continue
    choice = input(f"\n🤔 Do you want to set up GitHub for this project? (y/n): ").strip().lower()
    if choice != 'y':
        print("👋 GitHub setup cancelled")
        return
    
    # Setup steps
    steps = [
        ("Git Repository Setup", setup_git_repo),
        ("Initial Commit", create_initial_commit),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        if not step_func():
            print(f"❌ Failed at: {step_name}")
            return
    
    # GitHub remote setup
    print(f"\n{'='*20} GitHub Remote Setup {'='*20}")
    result = setup_github_remote()
    if not result:
        print("❌ Failed to set up GitHub remote")
        return
    
    success, repo_url = result
    if not success:
        return
    
    # Push to GitHub
    print(f"\n{'='*20} Push to GitHub {'='*20}")
    if not push_to_github():
        print("❌ Failed to push to GitHub")
        return
    
    # Success message
    print("\n" + "="*60)
    print("🎉 SUCCESS! Your Healthcare LLM project is now on GitHub!")
    print("="*60)
    print(f"🔗 Repository URL: {repo_url}")
    print(f"📂 Repository contains:")
    print(f"   ✅ Complete LoRA fine-tuning pipeline")
    print(f"   ✅ 2,309 training samples with 866 medical conditions")
    print(f"   ✅ 910 test samples for evaluation")
    print(f"   ✅ BioGPT-Large model recommendations")
    print(f"   ✅ Interactive testing and batch processing")
    print(f"   ✅ Comprehensive documentation")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Visit your repository: {repo_url}")
    print(f"   2. Add repository description on GitHub")
    print(f"   3. Add topics/tags: machine-learning, healthcare, nlp, pytorch")
    print(f"   4. Share with the community!")

if __name__ == "__main__":
    main()
