import dlib
import cv2
from imutils import face_utils
import imutils
import numpy as np
import scipy.interpolate
from copy import deepcopy
from colordict import ColorDict
from PIL import Image
import argparse

# Remaining issues: cv.imshow() in get_mask, self.show=True does not display mask on second iteration
# Remaining issues: poor method for fitting nose ring to correct location on other images

# Delete token.pickle to use with your google account - request access from joeyhark@gmail.com

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filename', required=False,
                help='[Str] File name of image to import',
                default='nicolascage.jpg')
ap.add_argument('-l', '--lipstick', required=False, action='store_false',
                help='Passing this argument removes lipstick', default=True)
ap.add_argument('-e', '--eyeliner', required=False, action='store_false',
                help='Passing this argument removes eyeliner', default=True)
ap.add_argument('-b', '--blush', required=False, action='store_false',
                help='Passing this argument removes blush', default=True)
ap.add_argument('-n', '--nosering', required=False, action='store_false',
                help='Passing this argument removes nose ring', default=True)
ap.add_argument('--lipcolor', required=False, help='[Str] lipstick color', default='red')
ap.add_argument('--eyecolor', required=False, help='[Str] eyeliner color', default='black')
ap.add_argument('--blushcolor', required=False, help='[Str] blush color', default='red')
ap.add_argument('-s', '--showsteps', required=False, action='store_true',
                help='Passing this argument will show output images of each step. Press keyboard to continue through outputs', default=False)
args = vars(ap.parse_args())


class ApplyMakeup:
    def __init__(self, image, show_steps=False, lip_color='red', eyeliner_color='black', blush_color='red', colors=ColorDict()):
        self.image = cv2.imread(image)
        self.show_steps = show_steps
        self.lip_color = colors[lip_color]
        self.eyeliner_color = colors[eyeliner_color]
        self.blush_color = colors[blush_color]

        # Landmarks
        self.all = None
        self.lips = None
        self.right_eye = None
        self.left_eye = None

    def get_landmarks(self):
        # Initialize dlib face detector and facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # Load input image
        image = deepcopy(self.image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect face
        face = detector(gray, 1)[0]

        # Determine facial landmarks for the face region, convert landmark x-y coordinates to numpy array
        landmarks_pre = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks_pre)
        self.all = landmarks
        self.lips = landmarks[48:]
        self.right_eye = landmarks[42:48]
        self.left_eye = landmarks[36:42]

        if self.show_steps:
            # Iterate over x-y coordinates of facial landmarks and draw them on image
            for i, (x, y) in enumerate(landmarks):
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

            # Show output image with facial landmarks
            cv2.imshow('Landmarks', image)
            cv2.waitKey(0)

    def get_mask(self, region_pts):
        # Create binary mask of ROI
        mask = np.zeros_like(self.image)
        if len(region_pts) > 2:
            mask = cv2.fillPoly(mask, [region_pts], (255, 255, 255))
        else:
            cv2.circle(mask, region_pts, int(len(mask)/17), (255, 255, 255), -1)

        # Show mask
        if self.show_steps:
            cv2.imshow('Mask', mask)
            cv2.waitKey(0)
        return mask

    def color_lips(self):
        mask = self.get_mask(self.lips)
        lip_color = np.zeros_like(mask)
        lip_color[:] = self.lip_color[::-1]
        colored_mask = cv2.bitwise_and(mask, lip_color)

        # Blur colored mask for realism
        colored_mask = cv2.GaussianBlur(colored_mask, (7, 7), 10)

        # Combine colored mask with original
        self.image = cv2.addWeighted(self.image, 1, colored_mask, 0.4, 0)

        # Show combined image
        if self.show_steps:
            cv2.imshow('Colored Mask', self.image)
            cv2.waitKey(0)

    def eye_liner(self, eye):
        # Create image copy for combining
        overlay = self.image.copy()

        # Extract upper and lower sections of eye landmarks
        upper = eye[:4]
        lower = np.vstack((eye[0], eye[3:]))

        # Initialize extrapolated point lists
        i_x_extrap = []
        i_y_extrap = []

        # Initialize counter
        j = 0

        # Iterate over upper and lower sections of eye landmarks
        for i in [upper, lower]:

            # Declare upper and lower sections as lists of x and y coordinates
            i_x = list(i[:, 0])
            i_y = list(i[:, 1])

            # Move landmarks away from the eye slightly for realism
            # j=0: upper, j=1: lower
            if j < 1:
                i_y[0] -= 5
                i_y[1] -= 4
                i_y[2] -= 4
                i_y[3] -= 5

                # Create a copy of y coordinates and move farther from original points
                i_y_new = deepcopy(i_y)
                i_y_new[1] -= 3
                i_y_new[2] -= 3
            else:
                i_y[0] += 1
                i_y[1] += 4
                i_y[2] += 4
                i_y[3] += 3
                i_y_new = deepcopy(i_y)
                i_y_new[1] += 3
                i_y_new[2] += 3

            # Interpolate curves from points
            curve_i = scipy.interpolate.interp1d(i_x, i_y, 'quadratic')
            curve_i_new = scipy.interpolate.interp1d(i_x, i_y_new, 'quadratic')

            # Create new list of x points and use curve function to obtain smoother list of points
            for point in np.arange(np.min(i_x), np.max(i_x)):
                i_x_extrap.append(point)
                i_y_extrap.append(int(curve_i(point)))
                i_x_extrap.append(point)
                i_y_extrap.append(int(curve_i_new(point)))

            # Add to counter to force into if statement for lower
            j += 1

        # Declare array and add extrapolated points back into pairs
        i_new = np.zeros((len(i_x_extrap), 2), np.int32)
        i_new[:, 0] = i_x_extrap
        i_new[:, 1] = i_y_extrap

        # Add points onto image copy
        for (x, y) in i_new:
            cv2.circle(overlay, (x, y), 2, self.eyeliner_color[::-1], -1)

        # Combine copy and original
        self.image = cv2.addWeighted(overlay, 0.3, self.image, 0.7, 0)

        # Show result
        if self.show_steps:
            cv2.imshow("Eye", self.image)
            cv2.waitKey(0)

    def blush(self, landmarks):
        # Manually chosen points to find a midpoint
        L_points = np.vstack((landmarks[48], landmarks[17]))
        R_points = np.vstack((landmarks[54], landmarks[26]))

        for i in [L_points, R_points]:
            # Find midpoint
            i_centroid = (int(np.sum(i[:, 0])/2), int(np.sum(i[:, 1])/2))

            # Get mask and color
            mask = self.get_mask(i_centroid)
            blush_color = np.zeros_like(mask)
            blush_color[:] = self.blush_color[::-1]

            # Combine, blur, combine
            colored_mask = cv2.bitwise_and(mask, blush_color)
            colored_mask = cv2.GaussianBlur(colored_mask, (77, 77), 1400)
            self.image = cv2.addWeighted(self.image, 1, colored_mask, 0.4, 0)

        # Show combined image
        if self.show_steps:
            cv2.imshow('Colored Mask', self.image)
            cv2.waitKey(0)

    def nose_ring(self, nose_ring_img):
        # Center nostril points
        nose_pts = (self.all[32], self.all[34])

        # Difference between points for image resize
        dx = nose_pts[1][0] - nose_pts[0][0]
        dy = nose_pts[0][1] - nose_pts[1][1]

        # Convert main image to PIL
        pil_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)

        # Load nose ring image in PIL to maintain alpha channel
        nose_ring = Image.open(nose_ring_img)

        # Convert to array for resize and rotate
        nose_ring = np.array(nose_ring)
        resized = imutils.resize(nose_ring, width=dx)
        rotated = imutils.rotate_bound(resized, -np.degrees(np.arctan2(dy, dx)))

        # Convert back to PIL for combining
        nose_ring = Image.fromarray(rotated)

        # Combine
        pil_image.paste(nose_ring, (nose_pts[0][0], nose_pts[0][1]-15), nose_ring)

        # Convert back to array and switch to BGR
        self.image = np.array(pil_image)
        self.image = self.image[:, :, ::-1]

        # Show
        if self.show_steps:
            cv2.imshow('Nose Ring', self.image)
            cv2.waitKey(0)


if __name__ == '__main__':

    # Initialize makeup image object
    makeup_image = ApplyMakeup(args['filename'],
                               lip_color=args['lipcolor'],
                               eyeliner_color=args['eyecolor'],
                               blush_color=args['blushcolor'],
                               show_steps=args['showsteps'])

    # Obtain facial landmarks
    makeup_image.get_landmarks()

    # Apply lipstick
    if args['lipstick']:
        makeup_image.color_lips()

    # Apply blush
    if args['blush']:
        makeup_image.blush(makeup_image.all)

    # Apply eyeliner
    if args['eyeliner']:
        makeup_image.eye_liner(makeup_image.right_eye)
        makeup_image.eye_liner(makeup_image.left_eye)

    # Apply nose ring
    if args['nosering']:
        makeup_image.nose_ring('nose_ring.png')

    # Write image to file
    cv2.imwrite('{}_with_makeup.jpg'.format(args['filename'].split('.')[0]), makeup_image.image)

    # Show final output
    print('[INFO] Displaying final output...')
    cv2.imshow("Final output - Exit with keystroke", makeup_image.image)
    cv2.waitKey(0)
