#!/usr/bin/env python3
"""
Cleanup Unused Files - Remove redundant and outdated files
Keep only essential files for clean project structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def analyze_files():
    """
    Analyze files in mbd2 directory and categorize them
    """
    print("🔍 ANALYZING FILES IN MBD2 DIRECTORY")
    print("=" * 60)
    
    mbd2_dir = Path("mbd2")
    
    # Define file categories
    essential_files = {
        # Core documentation
        "README.md": "Main documentation",
        "FILE_INDEX.md": "File catalog",
        
        # Best/Final models (keep only the best ones)
        "advanced_eeg_model_best.pth": "Best enhanced model (7.1% val acc)",
        "eeg_contrastive_400epochs_best.pth": "Best 400-epoch model (5.9% val acc)",
        
        # Core architecture (latest versions)
        "enhanced_eeg_transformer.py": "Enhanced EEG transformer architecture",
        "advanced_loss_functions.py": "Advanced loss functions",
        "data_augmentation.py": "Data augmentation pipeline",
        
        # Main training scripts (latest versions)
        "advanced_training.py": "Advanced training script (recommended)",
        "train_400_epochs.py": "Extended training script",
        
        # Essential utilities
        "monitor_training.py": "Training monitoring",
        "test_encoder_with_real_data.py": "Testing framework",
        
        # Data files (essential)
        "correctly_preprocessed_eeg_data.pkl": "Preprocessed EEG data",
        "explicit_eeg_data_splits.pkl": "Train/Val/Test splits",
        
        # Best training results
        "advanced_training_results_20250622_121116.png": "Advanced training curves",
        "eeg_training_400epochs_curves.png": "400-epoch training curves"
    }
    
    # Files to remove (redundant/outdated)
    files_to_remove = []
    
    # Scan all files
    all_files = list(mbd2_dir.glob("*"))
    
    print(f"📊 Found {len(all_files)} files total")
    print(f"📋 Essential files: {len(essential_files)}")
    
    # Categorize files
    for file_path in all_files:
        if file_path.is_file():
            filename = file_path.name
            
            if filename in essential_files:
                print(f"   ✅ KEEP: {filename} - {essential_files[filename]}")
            else:
                # Determine if file should be removed
                should_remove = False
                reason = ""
                
                # Remove old checkpoints (keep only best models)
                if filename.endswith('.pth') and 'checkpoint' in filename:
                    should_remove = True
                    reason = "Intermediate checkpoint (redundant)"
                
                # Remove old/redundant model files
                elif filename.endswith('.pth') and filename not in essential_files:
                    if 'explicit_eeg_contrastive_encoder' in filename:
                        should_remove = True
                        reason = "Superseded by enhanced model"
                    elif 'final' in filename:
                        should_remove = True
                        reason = "Final model (best model preferred)"
                
                # Remove old training plots (keep only latest/best)
                elif filename.endswith('.png') and filename not in essential_files:
                    should_remove = True
                    reason = "Old/redundant training plot"
                
                # Remove old/redundant scripts
                elif filename.endswith('.py') and filename not in essential_files:
                    if filename in ['explicit_eeg_contrastive_training.py', 
                                  'eeg_transformer_encoder.py',
                                  'correct_eeg_preprocessing_pipeline.py',
                                  'organize_files.py']:
                        should_remove = True
                        reason = "Superseded by enhanced versions"
                
                if should_remove:
                    files_to_remove.append((file_path, reason))
                    print(f"   ❌ REMOVE: {filename} - {reason}")
                else:
                    print(f"   ❓ REVIEW: {filename} - Unknown file")
    
    return essential_files, files_to_remove

def create_backup():
    """
    Create backup of files before deletion
    """
    print(f"\n💾 CREATING BACKUP")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"mbd2_backup_{timestamp}")
    
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    # Copy entire mbd2 directory
    shutil.copytree("mbd2", backup_dir)
    
    print(f"✅ Backup created: {backup_dir}")
    print(f"   You can restore files from here if needed")
    
    return backup_dir

def remove_files(files_to_remove, create_backup_flag=True):
    """
    Remove specified files
    """
    print(f"\n🗑️  REMOVING UNUSED FILES")
    print("=" * 60)
    
    if create_backup_flag:
        backup_dir = create_backup()
    
    removed_count = 0
    total_size_mb = 0
    
    for file_path, reason in files_to_remove:
        try:
            size_mb = file_path.stat().st_size / 1024 / 1024
            file_path.unlink()
            
            print(f"   ✅ Removed: {file_path.name} ({size_mb:.1f}MB) - {reason}")
            removed_count += 1
            total_size_mb += size_mb
            
        except Exception as e:
            print(f"   ❌ Failed to remove {file_path.name}: {e}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   🗑️  Files removed: {removed_count}")
    print(f"   💾 Space freed: {total_size_mb:.1f}MB")
    
    return removed_count, total_size_mb

def update_file_index():
    """
    Update FILE_INDEX.md after cleanup
    """
    print(f"\n📋 UPDATING FILE INDEX")
    print("=" * 60)
    
    mbd2_dir = Path("mbd2")
    index_file = mbd2_dir / "FILE_INDEX.md"
    
    # Collect file information
    files_info = []
    
    for file_path in sorted(mbd2_dir.iterdir()):
        if file_path.is_file() and file_path.name != "FILE_INDEX.md":
            size_mb = file_path.stat().st_size / 1024 / 1024
            
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
                'size_mb': size_mb
            })
    
    # Write updated index file
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write("# MBD2 File Index (Cleaned)\n\n")
        f.write("This directory contains essential files for EEG contrastive learning experiments.\n")
        f.write("Redundant and outdated files have been removed for clarity.\n\n")
        
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
        f.write(f"- Last cleaned: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ File index updated: {index_file}")

def show_final_structure():
    """
    Show final clean structure
    """
    print(f"\n📁 FINAL CLEAN STRUCTURE")
    print("=" * 60)
    
    mbd2_dir = Path("mbd2")
    
    # Group files by category
    categories = {
        "Documentation": [],
        "Model Files": [],
        "Script Files": [],
        "Data Files": [],
        "Plot Files": []
    }
    
    for file_path in sorted(mbd2_dir.iterdir()):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / 1024 / 1024
            
            if file_path.suffix == ".md":
                categories["Documentation"].append((file_path.name, size_mb))
            elif file_path.suffix == ".pth":
                categories["Model Files"].append((file_path.name, size_mb))
            elif file_path.suffix == ".py":
                categories["Script Files"].append((file_path.name, size_mb))
            elif file_path.suffix == ".pkl":
                categories["Data Files"].append((file_path.name, size_mb))
            elif file_path.suffix == ".png":
                categories["Plot Files"].append((file_path.name, size_mb))
    
    total_files = 0
    total_size = 0
    
    for category, files in categories.items():
        if files:
            print(f"\n📂 {category} ({len(files)} files):")
            for filename, size_mb in files:
                print(f"   - {filename} ({size_mb:.1f}MB)")
                total_files += 1
                total_size += size_mb
    
    print(f"\n📊 Final Summary:")
    print(f"   📁 Total files: {total_files}")
    print(f"   💾 Total size: {total_size:.1f}MB")
    print(f"   🎯 Structure: Clean and organized")

def main():
    """
    Main cleanup function
    """
    print("🧹 MBD2 FILE CLEANUP UTILITY")
    print("=" * 70)
    print("This script will remove redundant and outdated files")
    print("to keep only essential files for clean project structure.")
    print("=" * 70)
    
    # Step 1: Analyze files
    essential_files, files_to_remove = analyze_files()
    
    if not files_to_remove:
        print(f"\n✅ No files to remove - directory is already clean!")
        return
    
    print(f"\n⚠️  CLEANUP PLAN:")
    print(f"   📁 Files to keep: {len(essential_files)}")
    print(f"   🗑️  Files to remove: {len(files_to_remove)}")
    
    # Ask for confirmation
    response = input(f"\n❓ Proceed with cleanup? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        # Step 2: Remove files
        removed_count, freed_mb = remove_files(files_to_remove)
        
        # Step 3: Update file index
        update_file_index()
        
        # Step 4: Show final structure
        show_final_structure()
        
        print(f"\n🎉 CLEANUP COMPLETED!")
        print(f"   🗑️  Removed {removed_count} files")
        print(f"   💾 Freed {freed_mb:.1f}MB space")
        print(f"   📁 Project structure is now clean and organized")
        
    else:
        print(f"\n❌ Cleanup cancelled by user")

if __name__ == "__main__":
    main()
