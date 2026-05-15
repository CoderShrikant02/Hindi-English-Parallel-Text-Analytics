#!/usr/bin/env python3
import os
import shutil

os.chdir(r'c:\Users\Acer\Downloads\SMA IA')

# Backup old corrupted files
if os.path.exists('app.py'):
    shutil.move('app.py', 'app_old_corrupted.py')
    print("✓ Backed up old app.py")

if os.path.exists('wordcloud_ui.py'):
    shutil.move('wordcloud_ui.py', 'wordcloud_ui_old_corrupted.py')
    print("✓ Backed up old wordcloud_ui.py")

# Move new clean files to proper names
shutil.move('app_new.py', 'app.py')
print("✓ Installed new app.py")

shutil.move('wordcloud_ui_new.py', 'wordcloud_ui.py')
print("✓ Installed new wordcloud_ui.py")

print("\n✅ All files updated with clean English text!")
