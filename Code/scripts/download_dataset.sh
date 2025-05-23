#! /usr/bin/env sh

set -e

# Fetch and assemble the dataset if we don't have the full zip
if [ ! -f "PodcastFillers-full.zip" ]; then
    # Download the files for PodcastFillers from the hosting service.
    wget https://zenodo.org/records/7121457/files/PodcastFillers.csv
    wget https://zenodo.org/records/7121457/files/PodcastFillers.z01
    wget https://zenodo.org/records/7121457/files/PodcastFillers.z02
    wget https://zenodo.org/records/7121457/files/PodcastFillers.z03
    wget https://zenodo.org/records/7121457/files/PodcastFillers.zip

    # Assemble the files into a single record
    zip -FF PodcastFillers.zip --out PodcastFillers-full.zip

    # Clean up downloaded zips
    rm PodcastFillers.*
fi

# Unpack the dataset
unzip PodcastFillers-full.zip

# Move the dataset into Code/data
mkdir ../data
mv PodcastFillers ../data/.
