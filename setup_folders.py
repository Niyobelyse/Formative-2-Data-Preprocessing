"""
Setup script to create all necessary folders for the project
Run this FIRST before anything else
"""

import os

def create_folder_structure():
    """Create all necessary folders for the project"""
    
    folders = [
        'data/raw',
        'data/processed',
        'images/original',
        'images/augmented',
        'audio/original',
        'audio/augmented',
        'outputs/plots',
        'outputs/logs',
        'notebooks',
        'scripts'
    ]
    
    print("=" * 70)
    print("CREATING PROJECT FOLDER STRUCTURE")
    print("=" * 70)
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ Created: {folder}/")
    
    # Create __init__.py for scripts folder
    init_file = 'scripts/__init__.py'
    with open(init_file, 'w') as f:
        f.write("# Scripts module\n")
    print(f"✓ Created: {init_file}")
    
    print("\n" + "=" * 70)
    print("✅ FOLDER STRUCTURE CREATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Place your 3 images in: images/original/")
    print("   - neutral_1.jpg")
    print("   - smiling_1.jpg")
    print("   - surprised_1.jpg")
    print("\n2. Place your 2 audio files in: audio/original/")
    print("   - yes_approve.wav")
    print("   - confirm_transaction.wav")
    print("\n3. Run: pip install -r requirements.txt")
    print("4. Run: python scripts/run_all.py")
    print("=" * 70)

if __name__ == "__main__":
    create_folder_structure()