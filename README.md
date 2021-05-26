# Makeup Applier

**NOTE:** Requires Google API client permissions for use. Available upon request.

The goal of this project is to automatically apply makeup to an image of a face. The image is first obtained from Google Drive using the Google API client. This task was solved through direct use of a [tutorial](https://www.thepythoncode.com/article/using-google-drive--api-in-python). Once the image is present in the working directory, the makeup applier optionally applies lipstick, blush, eyeliner, and a nose ring to the face depending on user input.

Image editing is a common tool used in content creation, and automated facial image editing is prevalent on photography applications like Snapchat. This project aims to automate facial image editing on a single image using Dlib, OpenCV, and Python. An example result is shown below:

Without                    |  With
:-------------------------:|:-------------------------:
<img src="https://github.com/joeyhark/makeup_applier/blob/main/images/nicolascage.jpeg" width="300">  |  <img src="https://github.com/joeyhark/makeup_applier/blob/main/images/nicolascage_with_makeup.jpg" width="300">

The project is currently functional on a single image. The next stage will be adapting the program to run in real time, constantly applying makeup to a live video.

**Contents:**  
*Functional, In Use*  
`import_image.py` - pulls requested image from Google Drive to working directory  
`apply_makeup.py` - contains image makeup applier class, applies makeup to requested file based on user input  
`shape_predictor_68_face_landmarks.dat` - facial landmark detection algorithm based on [Kazemi and Sullivan (2014)](https://www.semanticscholar.org/paper/One-millisecond-face-alignment-with-an-ensemble-of-Kazemi-Sullivan/d78b6a5b0dcaa81b1faea5fb0000045a62513567?p2df)  
`nose_ring.png` - nose ring image for makeup applier  

*Non-Functional*   
images - sample images

**Issues/Improvements:**  
`apply_makeup.py`  
- [ ] cv.imshow() in get_mask, self.show=True does not display mask on second iteration.
- [ ] Poor method for fitting nose ring to correct location on other images.  

`import_image.py`  
- [ ] Filename display output incorrect.
- [ ] File size display output has no unit and may be unnecessary.
