#!/usr/bin/env python3
"""Comprehensive fix using only byte operations"""

import os

os.chdir(r'c:\Users\Acer\Downloads\SMA IA')

def fix_bytes(filepath):
    """Fix encoding by working with bytes directly"""
    
    # Read as bytes to avoid encoding issues
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Common mojibake byte sequences and their fixes
    # Using hex representation to avoid encoding issues
    fixes = [
        (b'\xc3\x82\xc2\xb7', b'\xc2\xb7'),      # Â· -> ·
        (b'\xc3\xb7', b'\xc3\xb7'),              # Ã·
        (b'\xc3\x97', b'\xc3\x97'),              # Ã—
        (b'\xc2\xba', b''),                      # Remove stray bytes
        (b'\xc3\xa9', b'\xc3\xa9'),              # é
    ]
    
    for old, new in fixes:
        data = data.replace(old, new)
    
    # Write back as UTF-8
    with open(filepath, 'wb') as f:
        f.write(data)

print("[PASS 2] Running byte-level encoding fix...\n")

files_to_fix = ['app.py', 'wordcloud_ui.py']

for filepath in files_to_fix:
    print(f"Processing {filepath}...")
    try:
        fix_bytes(filepath)
        print(f"  [OK] Fixed")
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n[SUCCESS] Encoding fixes complete!")
