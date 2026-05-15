#!/usr/bin/env python3
import re

print("Processing app.py...")
# Read the file with UTF-8 encoding
with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Remove the entire Word Cloud section
pattern = r'elif page == "Word Cloud Experiment":.*?(?=elif page|# FOOTER)'
content = re.sub(pattern, '', content, flags=re.DOTALL)

# Fix encoding issues
content = content.replace('â"€', '–')
content = content.replace('â€"', '–')
content = content.replace('â€"', '-')
content = content.replace('Â·', '·')
content = content.replace('Ã·', '÷')
content = content.replace('Ã—', '×')

# Write back with proper UTF-8 encoding
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Word Cloud section removed")
print("✓ Encoding issues fixed")

print("\nProcessing wordcloud_ui.py...")
# Similarly fix wordcloud_ui.py if needed
with open('wordcloud_ui.py', 'r', encoding='utf-8', errors='replace') as f:
    content_wc = f.read()

content_wc = content_wc.replace('â€"', '–')
content_wc = content_wc.replace('Â·', '·')

with open('wordcloud_ui.py', 'w', encoding='utf-8') as f:
    f.write(content_wc)

print("✓ wordcloud_ui.py fixed")
