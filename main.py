import tkinter
from tkinter import filedialog

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.



def importVideo():
   # faceDetector(mode='computerCamera')
    print_hi('video')


def importImage():
   # faceDetector(mode='videoFile')
    print_hi('image')

if __name__ == '__main__':
    top=tkinter.Tk()
    top.title('Licence Detection')
    top.geometry("220x150")
    button1 = tkinter.Button(top,text= "Import Video",command=importVideo)
    button1.place(relx=0.5, rely=0.3, anchor='s')
    button2 = tkinter.Button(top,text= "Import Image",command=importImage)
    button2.place(relx=0.5, rely=0.6, anchor='s')
    top.mainloop()
   # frame_video_convert.image_seq_to_video('my_data/frames_face_detector', output_path='./vid2vid.mp4', fps=15.0)










