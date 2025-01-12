import keyboard
import pyscreenshot
import numpy as np
import cv2
import easyocr
import time
import dxcam
import pyautogui

TYPING_SPEED = 0.11 # Default: 0.13
MAX_WORD_LENGTH = 11 # Default: 11
TURN_CHECK_DELAY = 0.7 # Default: 0.7
JOIN_GAME_DELAY = 15 # Default: 18
WORD_LIST = "scrabble_list.txt" # Default: "scrabble_list.txt"

reader = easyocr.Reader(['en'])
camera = dxcam.create()
last_turn_check = 0
join_game_timer = 0
is_turn = False

screen_width, screen_height = pyautogui.size()

with open(WORD_LIST) as word_file:
    valid_words = set(word_file.read().split())

# Find the longest valid word that matches as a subsequence
def find_longest_word(characters, dictionary, char_limit=16):
    longest_word = ""
    for word in dictionary:
        if characters in word and len(word) > len(longest_word) and len(word) <= char_limit:
            longest_word = word

    if longest_word != "":  # Remove the longest word from the dictionary
        dictionary.remove(longest_word)
        return longest_word
    else:
        return ""

def fix_text(text: str):
    chars = {'|': 'I', 
             '0': 'O', 
             '1': 'I', 
             '2': 'Z', 
             '3': 'B', 
             '4': 'A', 
             '5': 'S', 
             '6': 'G', 
             '7': 'T', 
             '8': 'B', 
             '9': 'G', 
             '$': 'S'}
    for key, value in chars.items():
        text = text.replace(key, value)
    
    if not text.isalpha() and text != "":
        print(text)
    text = "".join([char for char in text if char.isalpha()])

    return text

def my_preprocess(img, target_color, threshold):
    img_float = img.astype(np.float32)
    target_color = np.array(target_color, dtype=np.float32)
    distances = np.linalg.norm(img_float - target_color, axis=-1)
    mask = distances > threshold
    img[mask] = [255, 255, 255]
    img[~mask] = [0, 0, 0]

    return img

def stack_preprocess_1(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return img

def stack_preprocess_2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return thresh

def stack_preprocess_3(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)

    return thresh

def get_text_from_processed_img(img):
    output = ""
    for result in reader.readtext(img):
        coods, text, confidence = result
        output += "".join(text.split())

    return fix_text(output)

def read_text(img):
    my_img = my_preprocess(img, target_color=[0, 0, 0], threshold=50)
    stack_img = stack_preprocess_1(img)
    stack_img_2 = stack_preprocess_2(img)
    stack_img_3 = stack_preprocess_3(img)

    outputs = [get_text_from_processed_img(my_img), 
               get_text_from_processed_img(stack_img),
               get_text_from_processed_img(stack_img_2)]
    
    print(outputs)

    # remove duplicates
    outputs = list(set(outputs))

    print(outputs)

    # remove words with more than 3 characters
    outputs = [text for text in outputs if len(text) < 4]

    # longest set of characters comes first
    outputs.sort(key=lambda x: len(x), reverse=True)

    print(outputs)

    return outputs[0] if len(outputs) > 0 else ""

def enter_text(text, speed=TYPING_SPEED):
    keyboard.write(text, speed)
    keyboard.press('enter')

def check_is_turn(threshold=60):
    try:
        img = camera.grab((int(screen_width * 0.39), 
                           int(screen_height * 0.94), 
                           int(screen_width * 0.63), 
                           int(screen_height * 0.97))) # normalized because i use a 1440p monitor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # convert to black and white
        img_float = img.astype(np.float32)
        target_color = np.array([0, 0, 0], dtype=np.float32)
        distances = np.linalg.norm(img_float - target_color, axis=-1)
        mask = distances > threshold
        img[mask] = [255, 255, 255]
        img[~mask] = [0, 0, 0]

        return np.sum(img == 0) > 0.5 * img.size
    except Exception as e:
        print(e)
        check_is_turn(threshold)

while True:
    if time.time() - last_turn_check > TURN_CHECK_DELAY:
        is_turn = check_is_turn()
        last_turn_check = time.time()
    
    if time.time() - join_game_timer > JOIN_GAME_DELAY:
        pyautogui.click(int(screen_width * 0.44), int(screen_height * 0.56))
        join_game_timer = time.time()

    # if key m is pressed, do something
    if keyboard.is_pressed("esc") or is_turn:
        try:
            # take a screenshot of the screen
            bottom_img = np.array(pyscreenshot.grab(bbox=(int(screen_width * 0.4), 
                                                          int(screen_height * 0.62), 
                                                          int(screen_width * 0.48), 
                                                          int(screen_height * 0.65)))) # normalized
            top_img = np.array(pyscreenshot.grab(bbox=(int(screen_width * 0.4), 
                                                       int(screen_height * 0.36), 
                                                       int(screen_width * 0.48), 
                                                       int(screen_height * 0.39)))) # normalized

            bottom_img = bottom_img[:, :, ::-1].copy()
            top_img = top_img[:, :, ::-1].copy()

            bottom_text = read_text(bottom_img)
            top_text = read_text(top_img)

            characters = (bottom_text + top_text).lower()

            if len(characters) == 0:
                enter_text(find_longest_word("OL", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word("OO", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word("LO", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word("LL", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word("IL", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word("LI", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word("DI", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word("ID", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
            elif len(characters) == 1:
                enter_text(find_longest_word(characters.upper() + "I", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word(characters.upper() + "L", valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word("I" + characters.upper(), valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
                enter_text(find_longest_word("L" + characters.upper(), valid_words, char_limit=MAX_WORD_LENGTH), 0.04)
            else:
                longest_word = find_longest_word(characters.upper(), valid_words, char_limit=MAX_WORD_LENGTH)
                print(characters)
                print(longest_word)

                enter_text(longest_word)

            is_turn = False
            time.sleep(0.3)
        except Exception as e:
            print(e)
            pass

    if keyboard.is_pressed("`"):
        break