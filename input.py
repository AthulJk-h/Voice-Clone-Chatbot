from train import wishMe,takeCommand,Query
import tkinter
from tkinter import filedialog
from tkinter import *

def receive():
	input = filedialog.askopenfile(initialdir="/")
	print("embed")
def send():
	q = takeCommand()
	print(Query(q))
 

base = Tk()
base.title("Group 9 CHATBOT")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)


ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set
SendButton = Button(base, font=("Verdana",12,'bold'), text="VOICE", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
TargetVoice = Button(base, font=("Verdana",12,'bold'), text="BROWSE TARGETVOICE", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= receive)


scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
SendButton.place(x=6, y=401, height=45, width=370)
TargetVoice.place(x=6,y=447,height=45,width=370)
wishMe()
base.mainloop()
