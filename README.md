A simple self driving app for a motor boat.

Assuming you have a boat, and a river, here is how this works: you point your 
phone forward toward the river; it communicates via Bluetooth to a robotic arm connected to the steering 
gear and ensures the boat stays in the middle of the rover.

torch/
has the pytorch code to train a segmentation model, which will detect water from the banks 
and which is used by the android app. This includes a tool to extract frames - mkframes.py - from a video
stream and another tool - tagger.py - to actually tag images as water vs everything else.

android/
has the android app that parses what it sees via the camera and the torch netork and attempts
to keep the boat in the middle of the river.
