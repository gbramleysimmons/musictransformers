#!/bin/bash

#this downloads the zip file that contains the data
curl https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip --output midi.zip
# this unzips the zip file - you will get a directory named "data" containing the data
unzip midi.zip
# this cleans up the zip file, as we will no longer use it
rm midi.zip

mv maestro-v2.0.0 data

echo downloaded data
