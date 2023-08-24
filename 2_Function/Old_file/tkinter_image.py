import cv2
import socket
import tkinter as tk
import sys
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np


def cv2_to_tkinter_image(cv_image):
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    # size = (720, 720)
    # image = image.resize(size, Image.LANCZOS)
    image = ImageTk.PhotoImage(image)
    return image