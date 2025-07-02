import os
import shutil

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

UNWANTED_FOLDERS = [
    '__pycache__',
    'node_modules',
    '.ipynb_checkpoints'
]

UNWANTED_EXTENSIONS = [
    '.log', '.tmp', '.bak', '.pyc'
]

UNWANTED_FILES_BY_NAME = [
    'Thumbs.db', '.DS_Store', 'desktop.ini'
]


def delete_folder(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"[✓] Deleted folder: {path}")
        except Exception as e:
            print(f"[!] Failed to delete folder {path}: {e}")

def delete_file(path):
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"[✓] Deleted file: {path}")
        except Exception as e:
            print(f"[!] Failed to delete file {path}: {e}")

def clean_project(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        # حذف المجلدات الغير مهمة
        for dirname in dirnames:
            if dirname in UNWANTED_FOLDERS:
                full_path = os.path.join(dirpath, dirname)
                delete_folder(full_path)

        # حذف الملفات الغير مهمة
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            # حسب الامتداد
            if any(filename.endswith(ext) for ext in UNWANTED_EXTENSIONS):
                delete_file(file_path)

            # حسب الاسم الكامل
            if filename in UNWANTED_FILES_BY_NAME:
                delete_file(file_path)

def main():
    print(f"📂 Cleaning project at: {PROJECT_PATH}\n")
    clean_project(PROJECT_PATH)
    print("\n✅ Project cleaned successfully!")

if __name__ == "__main__":
    main()
