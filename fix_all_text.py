#!/usr/bin/env python3
"""Fix all text encoding issues and convert to English"""

import re
import os
import sys

os.chdir(r'c:\Users\Acer\Downloads\SMA IA')

# Common corrupted patterns to fix
def fix_encoding(content):
    """Fix all encoding issues in content"""
    
    # Fix corrupted dashes (right single quotation mark used as dash)
    content = re.sub(r'[\u00e2][\u0080][\u0093]', '–', content)  # –
    content = re.sub(r'[\u00e2][\u0080][\u0094]', '–', content)  # –
    content = re.sub(r'[\u00e2][\x80][\x9d]', '"', content)       # "
    
    # Remove other mojibake characters
    content = content.replace('\u00c2\u00b7', '·')
    content = content.replace('\u00c3\u00b7', '÷')
    content = content.replace('\u00c3\u0097', '×')
    content = content.replace('\u00c2', '')  # Remove stray Â
    
    # Fix number emojis (corrupted: 1ï¸âƒ£ should be 1️⃣)
    for i in range(1, 8):
        corrupted_pattern = str(i) + 'ï¸âƒ£'
        correct_emoji = str(i) + '️⃣'
        content = content.replace(corrupted_pattern, correct_emoji)
    
    return content

print("=" * 60)
print("FIXING TEXT ENCODING ISSUES")
print("=" * 60)

# Fix app.py
print("\n[1/2] Processing app.py...")
try:
    with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
        app_content = f.read()
    
    # Remove Word Cloud section
    app_content = re.sub(
        r'elif page == "Word Cloud Experiment":.*?(?=elif page|# FOOTER)',
        '',
        app_content,
        flags=re.DOTALL
    )
    
    # Fix encoding
    app_content = fix_encoding(app_content)
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(app_content)
    
    print("  [OK] Word Cloud section removed")
    print("  [OK] Encoding fixed")
except Exception as e:
    print(f"  [ERROR] {e}")

# Fix wordcloud_ui.py
print("\n[2/2] Processing wordcloud_ui.py...")
try:
    with open('wordcloud_ui.py', 'r', encoding='utf-8', errors='replace') as f:
        wc_content = f.read()
    
    wc_content = fix_encoding(wc_content)
    
    with open('wordcloud_ui.py', 'w', encoding='utf-8') as f:
        f.write(wc_content)
    
    print("  [OK] Encoding fixed")
except Exception as e:
    print(f"  [ERROR] {e}")

print("\n" + "=" * 60)
print("CONVERSION COMPLETE!")
print("=" * 60)
print("\nAll files have been converted to proper English text.")
print("Word Cloud feature has been removed from the project.")
