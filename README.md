# LAR_robot

**PRIORITY**
- get distance
- rotate
- detection half ball

**IDEA**

1. Create map
2. Look around
3. Calibration
4. Decesion tree

# ERRORS
- odometry loop
- image get (prob. bad robot)

# Classes

- **RobotControler**
    - safe robot movements - shutdown [x]
    - move by distance?
    - rotabe by angle?

- **ImageProcessor**
    - Classify rgb image
    - Add depth to image
    

# Threads

- **Main**
    - process queue
    - move robot

- **Image**
    - process images
    - classify objects

- **Exit**
    - bumper

- **GUI?**
    - console prints
    - show windows
- **DEPTH?**
    - get_distance