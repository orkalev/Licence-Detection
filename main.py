import cv2
import pytesseract
import tkinter
#from tkinter import filedialog
# Import module
from tkinter import *
import requests
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def importVideo():
   # faceDetector(mode='computerCamera')
    print_hi('video')


def importImage():
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        # Read the image file
        image = cv2.imread(path)
        # Convert to Grayscale Image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny Edge Detection
        canny_edge = cv2.Canny(gray_image, 170, 200)

        # Find contours based on Edges
        contours, new = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        # Initialize license Plate contour and x,y coordinates
        contour_with_license_plate = None
        license_plate = None
        x = None
        y = None
        w = None
        h = None

        # Find the contour with 4 potential corners and creat ROI around it
        for contour in contours:
            # Find Perimeter of contour and it should be a closed contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            if len(approx) == 4:  # see whether it is a Rect
                contour_with_license_plate = approx
                x, y, w, h = cv2.boundingRect(contour)
                license_plate = gray_image[y:y + h, x:x + w]
                break

        # Removing Noise from the detected image, before sending to Tesseract
        #cv2.imshow("License Plate Detection", license_plate)
        license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
        (thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)



        # Text Recognition
        #text = pytesseract.image_to_string(license_plate)
        text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 6')
        #text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 6')
        #text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')


        # Draw License Plate and write the Text
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        image = cv2.putText(image, text, (x - 100, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_AA)
       # cv2.createButton("Get Car Info", carInfo, None, cv2.QT_PUSH_BUTTON, 1)

        print("License Plate :", text)

        cv2.imshow("License Plate Detection", image)

        # #Get the data from API source
        payload = {'resource_id': '053cea08-09bc-40ec-8f7a-156f0677aff3', 'q': '5455354'}
        r = requests.get('https://data.gov.il/api/3/action/datastore_search', params=payload)
        res = r.json()
        record1 = res['result']['records']

        record = record1[0]
        mispar_rechev = record["mispar_rechev"]
        tozeret_cd = record["tozeret_cd"]
        degem_nm = record["degem_nm"]
        ramat_gimur = record["ramat_gimur"]
        ramat_eivzur_betihuty = record["ramat_eivzur_betihuty"]
        kvutzat_zihum = record["kvutzat_zihum"]
        shnat_yitzur = record["shnat_yitzur"]
        degem_manoa = record["degem_manoa"]
        mivchan_acharon_dt = record["mivchan_acharon_dt"]
        tokef_dt = record["tokef_dt"]
        baalut = record["baalut"]
        misgeret = record["misgeret"]
        tzeva_rechev = record["tzeva_rechev"]
        zmig_kidmi = record["zmig_kidmi"]
        zmig_ahori = record["zmig_ahori"]
        sug_delek_nm = record["sug_delek_nm"]
        horaat_rishum = record["horaat_rishum"]
        kinuy_mishari = record["kinuy_mishari"]

        print(kinuy_mishari)

#def carInfo():


def exitUI():
    # faceDetector(mode='videoFile')
    print_hi('exit')



# Create object
root = Tk()

# Adjust size
root.geometry("510x400")

root.title('Licence Detection')

# Add image file
bg = PhotoImage(file = "test2.png")

# Create Canvas
canvas1 = Canvas(root, width=510,
                 height=400)

canvas1.pack(fill="both", expand=True)

# Display image
canvas1.create_image(0, 0, image=bg,
                     anchor="nw")

# Add Text
canvas1.create_text(280, 30, text="Welcome to the licence detector")


# Create Buttons
importVideoButton = tkinter.Button(root,text= "Import Video",command=importVideo)
importImageButton = tkinter.Button(root,text= "Import Image",command=importImage)
exitButton = tkinter.Button(root,text= "Exit",command=exitUI)


# Display Buttons
importVideoButton_canvas = canvas1.create_window(30, 10,
                                       anchor="nw",
                                       window=importVideoButton)

importImageButton_canvas = canvas1.create_window(30, 40,
                                       anchor="nw",
                                       window=importImageButton)

exitButton_canvas = canvas1.create_window(30, 70, anchor="nw",
                                       window=exitButton)

# Execute tkinter
root.mainloop()

