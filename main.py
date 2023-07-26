import cv2
import colorama
from colorama import Fore, Style
import numpy as np
from numba import jit
import threading
import time
import os
from PIL import Image

colorama.init()

ASCII_CHARS = '@B%W#*oahkbdwmZO0QCJYXzcvnxrjft/\|()1{}[]-_+~<>i!lI;:,"^`\'.'

def bgr_to_rgb(bgr_color):
    return (bgr_color[2], bgr_color[1], bgr_color[0])

@jit(nopython=True)
def get_ascii_char(gray_val):
    index = int(gray_val / 255 * (len(ASCII_CHARS) - 1))
    return ASCII_CHARS[index]

def display_ascii_art(frame, fps):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = gray_frame.shape
    term_size = os.get_terminal_size()
    new_width = term_size.columns
    new_height = int(new_width * height / (width * 2))
    if new_height > term_size.lines - 1:
        new_height = term_size.lines - 1
        new_width = int(new_height * width * 2 / height)
    resized_frame = cv2.resize(gray_frame, (new_width, new_height))

    ascii_colors = np.empty_like(resized_frame, dtype=object)
    for y in range(new_height):
        for x in range(new_width):
            gray_val = resized_frame[y, x]
            char = get_ascii_char(gray_val)
            bgr_color = frame[int(y * height / new_height), int(x * width / new_width)]
            rgb_color = bgr_to_rgb(bgr_color)
            try:
                ascii_color = Fore.rgb(*rgb_color)
            except AttributeError:
                r, g, b = rgb_color
                ascii_color = f"\033[38;2;{r};{g};{b}m"

            ascii_colors[y, x] = f"{ascii_color}{char}{Style.RESET_ALL}"

    ascii_art = '\n'.join([''.join(row) for row in ascii_colors])
    os.system('cls' if os.name == 'nt' else 'clear')
    print(ascii_art)

    print(f"FPS: {fps:.2f}", end='\r')


def process_frames(cap):
    prev_time = time.time()
    fps = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        curr_time = time.time()
        time_diff = curr_time - prev_time
        fps = 1.0 / time_diff if time_diff > 0 else 0
        prev_time = curr_time
        display_ascii_art(frame, fps)

        cv2.waitKey(1)

def main():
    video_path = input('Video Path > ')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file.")
        return
    print('Opening Video...')
    frame_thread = threading.Thread(target=process_frames, args=(cap,))
    print('Starting frame thread...')
    frame_thread.start()
    print('Finished frame thread...')
    print('loading video...')

    frame_thread.join()

    cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
