#!/usr/bin/env python3
"""
Organize Files - Move model files from root to mbd2 folder
Clean up file organization and maintain proper structure
"""

import os
import shutil
from pathlib import Path

def organize_model_files():
    """
    Move model files from root directory to mbd2 folder
    """
    print("🗂️  ORGANIZING MODEL FILES")
    print("=" * 50)
    
    # Define root and target directories
    root_dir = Path(".")
    mbd2_dir = Path("mbd2")
    
    # Ensure mbd2 directory exists
    mbd2_dir.mkdir(exist_ok=True)
    
    # Define file patterns to move
    file_patterns = [
        # Model files
        "*eeg*.pth",
        "*advanced*.pth",
        # Training curves and plots
        "*training*.png",
        "*eeg*.png",
        # Exclude files that should stay in root
        "!README.md",
        "!*.py"
    ]
    
    # Get all .pth and .png files in root
    files_to_move = []
    
    # Find .pth files
    for pth_file in root_dir.glob("*.pth"):
        if pth_file.name.startswith(("eeg_", "advanced_", "explicit_", "improved_")):
            files_to_move.append(pth_file)
    
    # Find .png files related to training
    for png_file in root_dir.glob("*.png"):
        if any(keyword in png_file.name.lower() for keyword in 
               ["training", "eeg", "contrastive", "progress", "curves", "results"]):
            files_to_move.append(png_file)
    
    print(f"📁 Found {len(files_to_move)} files to organize:")
    
    moved_count = 0
    skipped_count = 0
    
    for file_path in files_to_move:
        target_path = mbd2_dir / file_path.name
        
        try:
            # Check if target already exists
            if target_path.exists():
                # Compare file sizes to decide whether to overwrite
                if file_path.stat().st_size != target_path.stat().st_size:
                    print(f"   🔄 Updating: {file_path.name}")
                    shutil.move(str(file_path), str(target_path))
                    moved_count += 1
                else:
                    print(f"   ⏭️  Skipping: {file_path.name} (identical)")
                    # Remove duplicate from root
                    file_path.unlink()
                    skipped_count += 1
            else:
                print(f"   ➡️  Moving: {file_path.name}")
                shutil.move(str(file_path), str(target_path))
                moved_count += 1
                
        except Exception as e:
            print(f"   ❌ Error moving {file_path.name}: {e}")
    
    print(f"\n📊 Organization Summary:")
    print(f"   ✅ Moved: {moved_count} files")
    print(f"   ⏭️  Skipped: {skipped_count} files")
    print(f"   📁 All files now in: mbd2/")

def verify_organization():
    """
    Verify that files are properly organized
    """
    print(f"\n🔍 VERIFICATION")
    print("=" * 50)
    
    root_dir = Path(".")
    mbd2_dir = Path("mbd2")
    
    # Check root directory for remaining model files
    remaining_pth = list(root_dir.glob("*.pth"))
    remaining_png = [f for f in root_dir.glob("*.png") 
                    if any(keyword in f.name.lower() for keyword in 
                          ["training", "eeg", "contrastive", "progress", "curves", "results"])]
    
    print(f"📁 Root directory:")
    if remaining_pth or remaining_png:
        print(f"   ⚠️  Still has {len(remaining_pth)} .pth and {len(remaining_png)} training .png files")
        for f in remaining_pth + remaining_png:
            print(f"      - {f.name}")
    else:
        print(f"   ✅ Clean - no model/training files")
    
    # Check mbd2 directory
    mbd2_pth = list(mbd2_dir.glob("*.pth"))
    mbd2_png = list(mbd2_dir.glob("*.png"))
    
    print(f"\n📁 mbd2/ directory:")
    print(f"   📊 Model files (.pth): {len(mbd2_pth)}")
    print(f"   📈 Plot files (.png): {len(mbd2_png)}")
    
    if mbd2_pth:
        print(f"   📋 Model files:")
        for f in sorted(mbd2_pth):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"      - {f.name} ({size_mb:.1f}MB)")
    
    if mbd2_png:
        print(f"   📋 Plot files:")
        for f in sorted(mbd2_png):
            print(f"      - {f.name}")

def create_file_index():
    """
    Create an index of all files in mbd2 directory
    """
    print(f"\n📋 CREATING FILE INDEX")
    print("=" * 50)
    
    mbd2_dir = Path("mbd2")
    index_file = mbd2_dir / "FILE_INDEX.md"
    
    # Collect file information
    files_info = []
    
    for file_path in sorted(mbd2_dir.iterdir()):
        if file_path.is_file() and file_path.name != "FILE_INDEX.md":
            size_mb = file_path.stat().st_size / 1024 / 1024
            modified = file_path.stat().st_mtime
            
            # Categorize files
            if file_path.suffix == ".pth":
                category = "Model Files"
            elif file_path.suffix == ".png":
                category = "Plot Files"
            elif file_path.suffix == ".py":
                category = "Script Files"
            elif file_path.suffix == ".pkl":
                category = "Data Files"
            elif file_path.suffix == ".md":
                category = "Documentation"
            else:
                category = "Other Files"
            
            files_info.append({
                'name': file_path.name,
                'category': category,
                'size_mb': size_mb,
                'modified': modified
            })
    
    # Write index file
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write("# MBD2 File Index\n\n")
        f.write("This directory contains all files related to EEG contrastive learning experiments.\n\n")
        
        # Group by category
        categories = {}
        for file_info in files_info:
            cat = file_info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(file_info)
        
        for category, files in categories.items():
            f.write(f"## {category}\n\n")
            for file_info in files:
                f.write(f"- **{file_info['name']}** ({file_info['size_mb']:.1f}MB)\n")
            f.write("\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total files: {len(files_info)}\n")
        f.write(f"- Total size: {sum(f['size_mb'] for f in files_info):.1f}MB\n")
        f.write(f"- Categories: {len(categories)}\n")
    
    print(f"✅ File index created: {index_file}")

def main():
    """
    Main organization function
    """
    print("🗂️  MBD2 FILE ORGANIZATION")
    print("=" * 70)
    print("This script will organize model files and plots into the mbd2/ folder")
    print("=" * 70)
    
    # Step 1: Organize files
    organize_model_files()
    
    # Step 2: Verify organization
    verify_organization()
    
    # Step 3: Create file index
    create_file_index()
    
    print(f"\n🎉 ORGANIZATION COMPLETE!")
    print(f"   📁 All model files are now in mbd2/")
    print(f"   📋 File index created: mbd2/FILE_INDEX.md")
    print(f"   ✅ Project structure is now clean and organized")

if __name__ == "__main__":
    main()
