# Makeup Applier

The goal of this project is to automatically apply makeup to an image of a face. The image is first obtained from Google Drive using the Google API client. This task was solved through direct use of a [tutorial](https://www.geeksforgeeks.org/upload-and-download-files-from-google-drive-storage-using-python/). Once the image is present in the working directory, the makeup applier optionally applies lipstick, blush, eyeliner, and a nose ring to the face depending on user input.

Image editing is a common tool used in content creation, and automated facial image editing is prevalent on photography applications like Snapchat. This project aims to automate facial image editing on a single image using Dlib, OpenCV, and Python. An example result is shown below:

Without                    |  With
:-------------------------:|:-------------------------:
<img src="https://github.com/joeyhark/makeup_applier/blob/main/images/nicolascage.jpeg" width="300">  |  <img src="https://github.com/joeyhark/makeup_applier/blob/main/images/nicolascage_with_makeup.jpg" width="300">

The project is currently functional on a single image. The next stage will be running the program in real time to constantly apply makeup to a live video.

**Contents:**  
*Functional, In Use*  
convenience  
&ensp;&ensp;`chromedriver` - executable used by WebDriver to control Chrome  
&ensp;&ensp;`results_convenience.py` - functions for calculating bet result and profit/loss from scraped match information  
&ensp;&ensp;`scrape_past.py` - function for scraping one day worth of matches that have already occured  
`main.py` - takes a date range and optional existing dataframe and yeilds a new dataframe with match information from given days  

*Semi-Functional, Obsolete*   
`notplayed.py` - yeilds dataframe with tomorrow's match information, leaving empty columns for match result, bet result, and profit/loss  
`results.py` - yeilds dataframe with results from played matches  

*Non-Functional*  
`combine.py` - aims to combine dataframes from notplayed.py and results.py and calculate bet result and profit/loss  
`test.py` - test script with changing uses  

**Issues/Improvements:**  
- [ ] Warn user if given dates are out of range (ongoing games, future games). Maintain functionality for capturing past games in date range after warning.
- [ ] Reduce explicit wait times to minimum without yielding errors, requires testing.
