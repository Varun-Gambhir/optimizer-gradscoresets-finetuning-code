#!/usr/bin/env python3
"""
Install packages from requirements.txt one by one.
If a package fails to install, it will be skipped and the next one will be installed.
"""

import subprocess
import sys

def install_packages_one_by_one(requirements_file="requirements.txt"):
    """
    Install packages from requirements.txt one by one.
    Skips packages that fail to install.
    """
    try:
        with open(requirements_file, 'r') as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: {requirements_file} not found!")
        sys.exit(1)
    
    total_packages = len(packages)
    successful = 0
    failed = 0
    skipped_packages = []
    
    print(f"Found {total_packages} packages to install.\n")
    print("=" * 70)
    
    for idx, package in enumerate(packages, 1):
        print(f"\n[{idx}/{total_packages}] Installing: {package}")
        print("-" * 70)
        
        try:
            # Run pip install with the exact package specification
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"✓ Successfully installed: {package}")
                successful += 1
            else:
                print(f"✗ Failed to install: {package}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
                failed += 1
                skipped_packages.append(package)
                
        except subprocess.TimeoutExpired:
            print(f"✗ Installation timed out for: {package}")
            failed += 1
            skipped_packages.append(package)
        except Exception as e:
            print(f"✗ Error installing {package}: {str(e)}")
            failed += 1
            skipped_packages.append(package)
    
    # Print summary
    print("\n" + "=" * 70)
    print("INSTALLATION SUMMARY")
    print("=" * 70)
    print(f"Total packages: {total_packages}")
    print(f"Successfully installed: {successful}")
    print(f"Failed/Skipped: {failed}")
    
    if skipped_packages:
        print("\nSkipped packages:")
        for pkg in skipped_packages:
            print(f"  - {pkg}")
    
    print("=" * 70)
    
    return successful, failed

if __name__ == "__main__":
    install_packages_one_by_one()
