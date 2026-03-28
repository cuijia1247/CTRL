# The codes are used to clear the temp folder
# Author: cuijia1247
# Date: 2014-1-6
# version: 1.0

import os
from tkinter import messagebox
import tkinter as tk

def clear_folder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_folder(file_path)
            os.rmdir(file_path)

if __name__ == '__main__':
    folderPath = './temp'
    root = tk.Tk()
    root.withdraw()
    root.after(3000, root.destroy)
    messagebox.showwarning("警告","确定要清空Temp文件夹吗？")
    clear_folder(folderPath)