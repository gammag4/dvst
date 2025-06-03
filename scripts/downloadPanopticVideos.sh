#!/bin/bash

# VGA VIEWS NOT SUPPORTED
# This script downloads <numVGAViews> VGA views and <numHDViews> HD views from
# the scenes listed in <panopticSceneNamesFile> into the path <downloadPath>
# and converts them to av1 with the constant quality of <cqAmount>
# ./downloadPanopticVideos.sh <panopticSceneNamesFile> <downloadPath> <numVGAViews> <numHDViews> <cqAmount>
#
# <panopticSceneNamesFile> should be a text file where each line has a scene

# ./downloadPanopticVideos.sh ../res/panoptic_scene_names.txt "/run/media/$USER/panoptic dataset 1" 0 8 30

panopticSceneNamesFile="$1"
downloadPath="$2"
numVGAViews=$3 #Specify the number of vga views you want to donwload. Up to 480
numHDViews=$4 #Specify the number of hd views you want to donwload. Up to 31
cqAmount=$5

scenes=($(cat "$panopticSceneNamesFile" | xargs))
mkdir -p "$downloadPath"

echo "Downloading $numVGAViews VGA views and $numHDViews HD views into $downloadPath"

vids=()
vidsProc=()
vidsUnproc=()
numVidsProc=()
numVidsUnproc=()
numVids=()

function getProcUnprocVids() {
    vidsPath="$1"

    # Listing total, processed and unprocessed videos
    vids=($(ls "$vidsPath"))
    vidsProc=()
    vidsUnproc=()
    for elem in ${vids[@]}; do
        if [[ $elem =~ .*_r\..* ]]; then
            vidsProc+=("$elem")
        else
            vidsUnproc+=("$elem")
        fi
    done

    # Counting videos
    numVidsProc="${#vidsProc[@]}"
    numVidsUnproc="${#vidsUnproc[@]}"
    numVids=$(expr $numVidsProc + $numVidsUnproc)
}

# Iterates over all scenes and downloads videos
for scene in ${scenes[@]}; do
    scenePath="$downloadPath/$scene"
    vidsPath="$scenePath/hdVideos"
    mkdir -p "$vidsPath"

    getProcUnprocVids "$vidsPath"

    # Downloads videos for the first time if not downloaded or again if incomplete
    if [[ $numVids -lt "$numHDViews" ]]; then
        echo "Downloading $scene"
        rm -r "$scenePath"
        ./getPanopticScene.sh "$scenePath" "$scene" "$numVGAViews" "$numHDViews"
    fi

    getProcUnprocVids "$vidsPath"

    # Converts videos to av1 format with lower quality
    for v in ${vidsUnproc[@]}; do
        echo "Processing videos in $scene"
        vname=$(echo $v | sed -E 's/\..*//g')
        vext=$(echo $v | sed -E 's/.*\.//g')
        # https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/ffmpeg-with-nvidia-gpu/index.html
        # ffmpeg -h encoder=av1_nvenc
        ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i "$vidsPath/$v" -map 0:v -c:v av1_nvenc -cq:v $cqAmount "$vidsPath/${vname}_r.$vext" \
            && rm "$vidsPath/$v"
    done
done
