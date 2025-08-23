cqAmount=23
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -vf scale_npp=640:-1 -map 0:v -c:v av1_nvenc -cq:v $cqAmount output.mp4
