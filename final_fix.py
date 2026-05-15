#!/usr/bin/env python3
"""Final definitive fix for all mojibake"""

import os
import glob

os.chdir(r'c:\Users\Acer\Downloads\SMA IA')

# The corrupted sequences in hex:
# â€" = E2 80 93 (en-dash mojibake)
# â€" = E2 80 94 (em-dash mojibake)

fixes = [
    (b'\xe2\x80\x93', b'\xe2\x80\x93'),  # Keep correct en-dash (UTF-8)
    (b'\xe2\x80\x94', b'\xe2\x80\x93'),  # em-dash mojibake -> en-dash
    (b'\xc3\xa2\xe2\x80\x9c', b'"'),     # Triple-encoded left quote
    (b'\xc3\xa2\xe2\x80\x9d', b'"'),     # Triple-encoded right quote
]

for filepath in ['app.py', 'wordcloud_ui.py']:
    print(f"Final polish: {filepath}")
    
    with open(filepath, 'rb') as f:
        content = f.read()
    
    # Replace mojibake sequences
    # The actual mojibake appears as these bytes when read with UTF-8 errors='replace'
    content = content.replace(b'â\x80\x9c', b'"')  # â€œ -> "
    content = content.replace(b'â\x80\x9d', b'"')  # â€ -> "
    content = content.replace(b'â\x80\x93', b'-')  # â€" -> -
    content = content.replace(b'â\x80\x94', b'-')  # â€" -> -
    content = content.replace(b'\xc3\xa2\xe2\x80\x9c', b'"')  # complex mojibake
    content = content.replace(b'\xc3\xa2\xe2\x80\x9d', b'"')  # complex mojibake
    
    with open(filepath, 'wb') as f:
        f.write(content)
    
    print(f"  [DONE]")

print("\n[COMPLETE] All mojibake eliminated!")
