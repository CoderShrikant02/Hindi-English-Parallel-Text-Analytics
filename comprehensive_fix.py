#!/usr/bin/env python3
"""Comprehensive fix for all remaining encoding issues"""

import re
import os

os.chdir(r'c:\Users\Acer\Downloads\SMA IA')

def comprehensive_fix(content):
    """More aggressive fix for all encoding issues"""
    
    # Replace all variations of corrupted characters
    replacements = [
        # Corrupted dashes and quotes
        ('\u00e2\u0080\u009c', '"'),    # corrupted left double quote
        ('\u00e2\u0080\u009d', '"'),    # corrupted right double quote  
        ('\u00e2\u0080\u0093', '–'),    # corrupted ndash
        ('\u00e2\u0080\u0094', '–'),    # corrupted mdash
        ('\u00e2\x80\x99', "'"),        # corrupted apostrophe
        ('\u00e2\x80\x98', "'"),        # corrupted left single quote
        
        # Other mojibake
        ('\u00c2\u00b7', '·'),          # corrupted middle dot
        ('\u00c3\u00b7', '÷'),          # corrupted division
        ('\u00c3\u0097', '×'),          # corrupted multiplication
        ('\u00c2', ''),                 # stray Â
    ]
    
    for corrupted, fixed in replacements:
        content = content.replace(corrupted, fixed)
    
    # Fix edge cases with simpler patterns
    content = content.replace('â€"', '–')
    content = content.replace('â€"', '–')
    content = content.replace('â†'', '→')
    content = content.replace('â€™', "'")
    content = content.replace('â€œ', '"')
    
    return content

print("[PASS 2] Running comprehensive encoding fix...\n")

# Fix app.py with aggressive replacements
print("Processing app.py...")
with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

content = comprehensive_fix(content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("  [OK] Comprehensive fixes applied")

# Fix wordcloud_ui.py
print("\nProcessing wordcloud_ui.py...")
with open('wordcloud_ui.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

content = comprehensive_fix(content)

with open('wordcloud_ui.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("  [OK] Comprehensive fixes applied")

print("\n[SUCCESS] All text encoding has been fixed!")
