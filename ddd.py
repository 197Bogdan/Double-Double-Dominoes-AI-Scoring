import cv2 as cv
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import multiprocessing

game_count = 5
input_path = 'input/'
templates_path = "templates/"
output_path = 'output/'
aux_path = 'auxiliary/'


fixed_width = 1200
fixed_height = 1200
square_count = 15
square_size = fixed_height // square_count
board_shape = (square_count, square_count)
vertical_lines = [x for x in range(0, fixed_width, fixed_width // square_count)]
horizontal_lines = [x for x in range(0, fixed_height, fixed_height // square_count)]


score_board = [[5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
               [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
               [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
               [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
               [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
               [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
               [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
               [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
               [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
               [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
               [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5]]

ladder_tiles = [-1, 1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0, 3, 5, 4, 1, 6, 2, 4, 5,
                5, 0, 6, 3, 4, 2, 0, 1, 5, 1, 3, 4, 4, 4, 5, 0, 6, 3, 5, 4,
                1, 3, 2, 0, 0, 1, 1, 2, 3, 6, 3, 5, 2, 1, 0, 6, 6, 5, 2, 1, 2,
                5, 0, 3, 3, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4, 2, 6, 2, 3, 1, 6, 5,
                6, 2, 0, 4, 0, 1, 6, 4, 4, 1, 6, 6, 3, 0]


def showImage(title, image):
    image = cv.resize(image, (0, 0), fx=0.7, fy=0.7)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def parseMoveFile(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        piece_placement, piece_type = lines[0].split()
        piece_placement2, piece_type2 = lines[1].split()
        score = lines[2]
        return (piece_placement, piece_placement2), [int(piece_type), int(piece_type2)], score


def indicesToBoardNotation(indices):
    letters = "ABCDEFGHIJKLMNO"
    piece_1 = f"{indices[0][0] + 1}{letters[indices[0][1]]}"
    piece_2 = f"{indices[1][0] + 1}{letters[indices[1][1]]}"
    return piece_1, piece_2


def cropScoreBoard(image):
    height, width, _ = image.shape
    margin_of_error = -0.015
    left = 0.159 + margin_of_error
    top = 0.1548 + margin_of_error
    right = 1 - (0.1527 + margin_of_error)
    bottom = 1 - (0.1548 + margin_of_error)
    left_corner = int(left * width)
    right_corner = int(right * width)
    top_corner = int(top * height)
    bottom_corner = int(bottom * height)
    return image[top_corner:bottom_corner, left_corner:right_corner]


def getFullBoard(image):
    kernel = np.ones((5, 5), np.uint8)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    dilated = cv.dilate(edges, kernel, iterations=3)
    eroded = cv.erode(dilated, kernel, iterations=1)

    contours, _ = cv.findContours(eroded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:1]
    x, y, w, h = cv.boundingRect(contours[0])
    cropped = image[y:y + h, x:x + w]

    return cropped


def getGameBoard(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh_black = cv.threshold(blurred, 30, 255, cv.THRESH_BINARY)
    _, thresh_white = cv.threshold(blurred, 220, 255, cv.THRESH_BINARY)
    thresh = thresh_white + (255 - thresh_black)

    kernel_size = 5
    iteration_count = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv.dilate(thresh, kernel, iterations=iteration_count)
    eroded = cv.erode(dilated, None, iterations=1)
    contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    for i in range(len(contours)):
        if len(contours[i]) > 3:
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    offset = -5  # handpicked
    dilated_width = (kernel_size - 1) * iteration_count + offset
    top_left = (top_left[0] + dilated_width, top_left[1] + dilated_width)
    bottom_right = (bottom_right[0] - dilated_width, bottom_right[1] - dilated_width)
    top_right = (top_right[0] - dilated_width, top_right[1] + dilated_width)
    bottom_left = (bottom_left[0] + dilated_width, bottom_left[1] - dilated_width)

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    resized_corners = np.array([[0, 0], [fixed_width, 0], [fixed_width, fixed_height], [0, fixed_height]],
                               dtype="float32")
    transform_matrix = cv.getPerspectiveTransform(corners, resized_corners)
    result = cv.warpPerspective(image, transform_matrix, (fixed_width, fixed_height))

    return result


def classifyPieceType(image, indices):
    templates = os.listdir(templates_path)
    patch1 = image[indices[0][0] * square_size:(indices[0][0] + 1) * square_size,
             indices[0][1] * square_size:(indices[0][1] + 1) * square_size]
    patch2 = image[indices[1][0] * square_size:(indices[1][0] + 1) * square_size,
             indices[1][1] * square_size:(indices[1][1] + 1) * square_size]

    patch1 = cv.cvtColor(patch1, cv.COLOR_BGR2GRAY)
    patch1 = cv.GaussianBlur(patch1, (5, 5), 0)
    _, patch1 = cv.threshold(patch1, 200, 255, cv.THRESH_BINARY)

    patch2 = cv.cvtColor(patch2, cv.COLOR_BGR2GRAY)
    patch2 = cv.GaussianBlur(patch2, (5, 5), 0)
    _, patch2 = cv.threshold(patch2, 200, 255, cv.THRESH_BINARY)

    all_rectangles_1 = []
    all_rectangles_2 = []
    w, h = 23, 23  # biggest template size
    for template in templates:
        template = templates_path + template
        template_image = cv.imread(template)
        template_image = cv.cvtColor(template_image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(template_image, (5, 5), 0)
        _, template_image = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
        res_1 = cv.matchTemplate(patch1, template_image, cv.TM_CCOEFF_NORMED)
        res_2 = cv.matchTemplate(patch2, template_image, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc_1 = np.where(res_1 >= threshold)
        loc_2 = np.where(res_2 >= threshold)

        rectangles_1 = [(*pt, w, h) for pt in zip(*loc_1[::-1])]
        rectangles_2 = [(*pt, w, h) for pt in zip(*loc_2[::-1])]
        all_rectangles_1.extend(rectangles_1)
        all_rectangles_2.extend(rectangles_2)
    indices_1 = cv.dnn.NMSBoxes(all_rectangles_1, [1.0] * len(all_rectangles_1), 0.5, 0.1)
    indices_2 = cv.dnn.NMSBoxes(all_rectangles_2, [1.0] * len(all_rectangles_2), 0.5, 0.1)
    prediction = [len(indices_1), len(indices_2)]
    prediction = [min(x, 6) for x in prediction]  # in case predictions exceeds max

    return prediction


def detectPiecePlacement(image1, image2):
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    scores = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            patch1 = gray1[h_line:h_line + square_size, v_line:v_line + square_size]
            patch2 = gray2[h_line:h_line + square_size, v_line:v_line + square_size]

            patch1 = cv.threshold(patch1, 150, 255, cv.THRESH_BINARY)[1]
            patch2 = cv.threshold(patch2, 150, 255, cv.THRESH_BINARY)[1]
            score = ssim(patch1, patch2)
            scores.append(score)

    scores = np.array(scores)
    indices = np.unravel_index(np.argsort(scores, axis=None)[:2], board_shape)
    points = []
    for x_coord, y_coord in zip(indices[0], indices[1]):
        points.append([x_coord, y_coord])
    sorted_points = sorted(points)
    return sorted_points
    # f = np.unravel_index(np.argsort(scores, axis=None)[:10], board_shape)     # top10 candidate patches
    # points_f = []
    # for x_coord, y_coord in zip(f[0], f[1]):
    #     points_f.append([x_coord, y_coord])


def writeOutput(img_file, output):
    file_name = output_path + img_file.split('.')[0] + '.txt'
    predicted_position, predicted_type, score = output
    with open(file_name, 'w') as f:
        f.write(f"{predicted_position[0]} {predicted_type[0]}\n")
        f.write(f"{predicted_position[1]} {predicted_type[1]}\n")
        f.write(f"{score}\n")


def worker(process_nr):
    print(f"Process {process_nr} started")
    files = [file for file in os.listdir(input_path) if file.startswith(f"{process_nr}")]
    image_files = [file for file in files if file.endswith('.jpg')]

    empty_game_board = cv.imread(aux_path + "01.jpg")
    empty_game_board = getFullBoard(empty_game_board)
    empty_game_board = cropScoreBoard(empty_game_board)
    empty_game_board = getGameBoard(empty_game_board)
    prev_game_board = empty_game_board

    with open(input_path + f"{process_nr}_mutari.txt", 'r') as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) > 3]
        turns = [line[-1] for line in lines]

    player1_score = 0
    player2_score = 0
    for img_file, turn in zip(image_files, turns):
        print(f"Working on {img_file}")
        if img_file.endswith("01.jpg"):
            prev_game_board = empty_game_board
        input_image = cv.imread(input_path + img_file)
        showImage("input", input_image)
        full_board = getFullBoard(input_image)
        showImage("full", full_board)
        crop = cropScoreBoard(full_board)
        showImage("crop", crop)
        game_board = getGameBoard(crop)
        showImage("game", game_board)
        indices = detectPiecePlacement(prev_game_board, game_board)
        predicted_position = indicesToBoardNotation(indices)
        predicted_type = classifyPieceType(game_board, indices)

        turn_score = 0
        if turn == '1':
            turn_score += score_board[indices[0][0]][indices[0][1]]
            turn_score += score_board[indices[1][0]][indices[1][1]]
            if predicted_type[0] == predicted_type[1]:
                turn_score = turn_score * 2
            if ladder_tiles[player1_score] in predicted_type:
                turn_score += 3
            if ladder_tiles[player2_score] in predicted_type:
                player2_score += 3
            player1_score += turn_score
        else:
            turn_score += score_board[indices[0][0]][indices[0][1]]
            turn_score += score_board[indices[1][0]][indices[1][1]]
            if predicted_type[0] == predicted_type[1]:
                turn_score = turn_score * 2
            if ladder_tiles[player1_score] in predicted_type:
                player1_score += 3
            if ladder_tiles[player2_score] in predicted_type:
                turn_score += 3
            player2_score += turn_score

        writeOutput(img_file, [predicted_position, predicted_type, turn_score])
        prev_game_board = game_board


if __name__ == "__main__":
    for i in range(1, game_count + 1):
        p = multiprocessing.Process(target=worker, args=(i,))
        p.start()
