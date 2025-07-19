"""
Healthcare LLM Setup Script
===========================
This script sets up the environment for healthcare LLM fine-tuning and testing.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_conda_env():
    """Check if we're in the correct conda environment."""
    try:
        result = subprocess.run("conda info --envs", shell=True, capture_output=True, text=True)
        current_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        print(f"📍 Current conda environment: {current_env}")
        
        if current_env != 'test_env':
            print(f"⚠️  Warning: You might want to activate the 'test_env' environment")
            print(f"Run: conda activate test_env")
        
        return True
    except Exception as e:
        print(f"❌ Error checking conda environment: {e}")
        return False

def install_requirements():
    """Install required packages."""
    print(f"\n🏥 Healthcare LLM Environment Setup")
    print(f"=" * 50)
    
    # Check conda environment
    check_conda_env()
    
    # Install PyTorch (with CUDA if available)
    print(f"\n🔥 Installing PyTorch...")
    pytorch_commands = [
        # Try CUDA 11.8 first
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        # Fallback to CPU-only
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    ]
    
    pytorch_installed = False
    for i, cmd in enumerate(pytorch_commands):
        if i == 0:
            desc = "Installing PyTorch with CUDA 11.8 support"
        else:
            desc = "Installing PyTorch (CPU-only fallback)"
            
        if run_command(cmd, desc):
            pytorch_installed = True
            break
    
    if not pytorch_installed:
        print(f"❌ Failed to install PyTorch")
        return False
    
    # Install Transformers and other ML libraries
    ml_packages = [
        ("pip install transformers", "Installing Hugging Face Transformers"),
        ("pip install datasets", "Installing Hugging Face Datasets"),
        ("pip install peft", "Installing PEFT (LoRA)"),
        ("pip install accelerate", "Installing Accelerate"),
    ]
    
    for cmd, desc in ml_packages:
        if not run_command(cmd, desc):
            print(f"⚠️  Warning: Failed to install {desc}")
    
    # Install data science packages
    data_packages = [
        ("pip install pandas numpy", "Installing data processing libraries"),
        ("pip install scikit-learn", "Installing scikit-learn"),
        ("pip install matplotlib seaborn", "Installing visualization libraries"),
    ]
    
    for cmd, desc in data_packages:
        if not run_command(cmd, desc):
            print(f"⚠️  Warning: Failed to install {desc}")
    
    # Install development tools
    dev_packages = [
        ("pip install jupyter ipykernel notebook", "Installing Jupyter tools"),
        ("pip install tqdm", "Installing progress bars"),
        ("pip install wandb", "Installing Weights & Biases (optional)"),
        ("pip install tensorboard", "Installing TensorBoard (optional)"),
    ]
    
    for cmd, desc in dev_packages:
        run_command(cmd, desc)  # These are optional
    
    # Try to install optional performance packages
    print(f"\n⚡ Installing optional performance packages...")
    optional_packages = [
        ("pip install bitsandbytes", "Installing BitsAndBytes for quantization"),
        # Flash attention requires CUDA and specific setup
        # ("pip install flash-attn", "Installing Flash Attention"),
    ]
    
    for cmd, desc in optional_packages:
        print(f"\n🔄 {desc} (optional)")
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ {desc} completed")
        except:
            print(f"⚠️  {desc} failed (optional)")
    
    return True

def verify_installation():
    """Verify that all packages are installed correctly."""
    print(f"\n🔍 Verifying installation...")
    
    packages_to_check = [
        "torch",
        "transformers", 
        "datasets",
        "peft",
        "sklearn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    failed_packages = []
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  The following packages failed to import: {failed_packages}")
        print(f"You may need to install them manually:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
        return False
    else:
        print(f"\n🎉 All packages installed successfully!")
        return True

def main():
    """Main setup function."""
    print(f"🏥 Healthcare LLM Environment Setup")
    print(f"🔧 This script will install all required dependencies")
    print(f"=" * 60)
    
    # Check if requirements.txt exists
    if os.path.exists("requirements.txt"):
        print(f"📋 Found requirements.txt")
        
        # Ask user for installation method
        choice = input(f"\nChoose installation method:\n1. Install step by step (recommended)\n2. Use requirements.txt directly\nEnter choice (1 or 2): ").strip()
        
        if choice == "2":
            if run_command("pip install -r requirements.txt", "Installing from requirements.txt"):
                verify_installation()
            return
    
    # Step by step installation
    if install_requirements():
        verify_installation()
        
        print(f"\n🎯 Setup complete! You can now run:")
        print(f"  python healthcare_lora_finetuning.py  # For training")
        print(f"  python healthcare_lora_testing.py    # For testing")
    else:
        print(f"\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
