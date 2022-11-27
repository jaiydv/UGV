from tkinter import *
from tkinter import ttk

#Create an instance of Tkinter frame
win= Tk()

#Set the geometry of Tkinter frame
win.geometry("750x250")

def display_text():
   global entry
   string= entry.get()
   label.configure(text=string)

#Initialize a Label to display the User Input
device=Label(win, text="", font=("Courier 22 bold"))
device.pack(pady=40)

#Initialize a Label to display the User Input
label=Label(win, text="Enter camera devide number", font=("Courier 22 bold"))
label.pack()

#Create an Entry widget to accept User Input
entry= Entry(win, width= 40)
entry.focus_set()
entry.pack()
print(entry.get())


#Create a Button to validate Entry Widget
ttk.Button(win, text= "Enter camera device number",width= 20,command=display_text).pack(pady=20)

win.mainloop()
