This is a bot that plays the Sigmar's Garden solitaire minigame in Opus Magnum.

The actual solver is a very simple depth-first search with memoization.

The interesting part is parsing the image.  My classifier is a nearest-centroid classifier based on four histograms: raw RGB values, Hough circle sizes, GFTT keypoint radial distribution about the center, and MSER keypoint radial distribution about the center.  RGB color obviously distinguishes the color of the atoms.  Circle detection helps differentiate between salt and quicksilver.  The two keypoint histograms can be seen as hacky feature descriptors; GFTT is very sensitive to the atom's symbol, while MSER is more sensitive to the non-symbol parts of the atom.  We take per-class averages of each histogram.  To classify an image, for each class, we take the sum of the "alternate" chi-squared distance score between the class's and the instance's histograms, and choose the class with the least score.  This is reasonably but not perfectly accurate.  The classifier also remembers its training data, so if you add mistakes to the training set, it will not make the same mistake twice.

You need numpy, OpenCV, Pillow and pyautogui to use this bot.

This bot assumes the Opus Magnum window is 1920x1080, so you probably need to have a 4K monitor to use it.  If you need to use a different window size, use find-grid.py to regenerate the constants in data.py.  find-grid.py takes as input a blank Sigmar's Garden board (as after solving) with other parts of the image, including the decorations in the corners around the board, replaced with a solid color (so as not to give rise to any circles or MSER keypoints).

To use this bot you also need some images in data/ which I do not distribute with this repo due to copyright reasons:

- data/upper-left.png: a picture of the upper-left corner of the Opus Magnum screen, whichever act you like, so long as you use that one to play the bot.
- data/classes/: directories of atom images to train the classifier.  There is one directory per class, and two classes per atom type: one class for active atoms (named `air` or `lead`, etc.) and one for inactive, grayed-out atoms (named `air-inactive` or `lead-inactive`, etc.).  You can use split-image.py to split screenshots into individual atom images that you can classify.

In practice, the bot still needs to be babysat a bit.  You can try adding a loop that presses the new game button and plays another game, but the bot will eventually stop when it misparses a puzzle, probably within just a few puzzles.
