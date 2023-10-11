import tkinter as tk
from tkinter import ttk
from tkinter import font

# Tkinter 창 생성
window = tk.Tk()
window.title("콤보박스 리스트 폰트 크기 조정 예제")

# 스타일 설정
style = ttk.Style()
style.configure('TCombobox', font=('Helvetica', 30))  # 폰트 크기 조정

# 콤보박스 생성
combo = ttk.Combobox(window, values=["옵션 1", "옵션 2", "옵션 3"])
combo.pack()

# 창 실행
window.mainloop()


import tkinter as tk
import tkinter.font as tkFont

root = tk.Tk()
root.geometry('300x200')

combobox = tkFont.Font(family='Helvetica', size=36)
options = [1,1,2,3,4,5]
selected = tk.StringVar(root, value="")

choose_test = tk.OptionMenu(root, selected, *options)
choose_test.config(font=combobox) # set the button font

combolist = tkFont.Font(family='Helvetica', size=20)
menu = root.nametowidget(choose_test.menuname)  # Get menu widget.
menu.config(font=combolist)  # Set the dropdown menu's font
choose_test.grid(row=0, column=0, sticky='nsew')

root.mainloop()