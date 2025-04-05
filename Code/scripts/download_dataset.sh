#! /usr/bin/env sh

# Fetch and assemble the dataset if we don't have the full zip
if [ -f "PodcastFillers-full.zip" ]; then
    # Download the files for PodcastFillers from the hosting service.
    wget https://zenodo.org/records/7121457/files/PodcastFillers.csv
    wget https://zenodo.org/records/7121457/files/PodcastFillers.z01
    wget https://zenodo.org/records/7121457/files/PodcastFillers.z02
    wget https://zenodo.org/records/7121457/files/PodcastFillers.z03
    wget https://zenodo.org/records/7121457/files/PodcastFillers.zip

    # Assemble the files into a single record
    zip -FF PodcastFillers.zip --out PodcastFillers-full.zip
fi

# Unpack the dataset
unzip PodcastFillers-full.zip

# Move the whole dataset up one directory
mv PodcastFillers ../.
