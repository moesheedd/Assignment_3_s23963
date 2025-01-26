import cv2
import numpy as np
from scipy.spatial import distance
import kociemba

# Define HSV ranges for Rubik's Cube colors
COLOR_RANGES = {
    "White": [(0, 0, 110), (180, 40, 225)],
    "Red_Upper": [(0, 117, 5), (6, 194, 100)],
    "Red_Lower": [(165, 17, 15), (180, 151, 106)],
    "Green": [(61, 62, 48), (94, 172, 160)],
    "Blue": [(98, 99, 0), (118, 199, 255)],
    "Yellow": [(19, 77, 100), (35, 156, 226)],
    "Orange": [(6, 37, 86), (12, 199, 159)],
}

CUBE_FACES = ["U", "L", "F", "R", "B", "D"]

COLOR_TO_FACE = {
    "White": "U",
    "Red_Upper": "R",
    "Red_Lower": "R",
    "Green": "F",
    "Blue": "B",
    "Yellow": "D",
    "Orange": "L",
}

def detect_color(hsv_color):
    """
    Detect the color of a given HSV value.
    """
    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        if cv2.inRange(np.uint8([[hsv_color]]), lower, upper):
            return color_name
    return "Unknown"

def cluster_centroids(centroids, distance_threshold=40):
    """
    Cluster centroids that are close to each other.
    """
    if not centroids:
        return []
    clustered_centroids = []
    visited = set()
    for i, c1 in enumerate(centroids):
        if i in visited:
            continue
        cluster = [c1]
        for j, c2 in enumerate(centroids):
            if j != i and j not in visited:
                if distance.euclidean(c1, c2) < distance_threshold:
                    cluster.append(c2)
                    visited.add(j)
        avg_cx = int(sum([c[0] for c in cluster]) / len(cluster))
        avg_cy = int(sum([c[1] for c in cluster]) / len(cluster))
        clustered_centroids.append((avg_cx, avg_cy))
    return clustered_centroids

def detect_stickers_within_grid(frame):
    """
    Detect centroids of the 9 stickers within the grid.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blur, 192, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=2)

    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    centroids = []

    for i, contour in enumerate(contours):
        if 800 < cv2.contourArea(contour) < 5000 and hierarchy[i][3] != -1:
            epsilon = 0.06 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    centroids.append((cx, cy))

    clustered_centroids = cluster_centroids(centroids)
    return sorted(clustered_centroids, key=lambda c: (c[1], c[0]))

def extract_colors(frame, centroids):
    """
    Extract detected colors for each centroid.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_colors = []

    for cx, cy in centroids:
        size = 20
        patch = hsv_frame[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2]
        if patch.size > 0:
            mean_color = np.mean(patch, axis=(0, 1)).astype(int)
            detected_color = detect_color(mean_color)
            detected_colors.append(COLOR_TO_FACE.get(detected_color, "X"))

    return detected_colors

def display_detection_data(frame, centroids, colors):
    """
    Display annotations on the frame for detected centroids and their colors.
    """
    for (cx, cy), color in zip(centroids, colors):
        cv2.putText(frame, color, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Unable to open the webcam!")
        return

    cube_faces = {}
    face_index = 0

    while face_index < 6:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break

        centroids = detect_stickers_within_grid(frame)
        colors = extract_colors(frame, centroids)

        if len(colors) == 9:
            annotated_frame = display_detection_data(frame.copy(), centroids, colors)
            cv2.imshow(f"Face {CUBE_FACES[face_index]}", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(colors) == 9:
            face_string = "".join(colors)
            cube_faces[CUBE_FACES[face_index]] = face_string
            print(f"Captured {CUBE_FACES[face_index]} face: {face_string}")
            face_index += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(cube_faces) == 6:
        # Construct the full cube string
        cube_string = "".join([cube_faces[face] for face in CUBE_FACES])
        print("Cube string:", cube_string)

        try:
            solution = kociemba.solve(cube_string)
            print("Solution:", solution)
            print("Number of moves:", len(solution.split()))
        except Exception as e:
            print("Error solving the cube:", str(e))


if __name__ == "__main__":
    main()