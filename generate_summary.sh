folder=$1
echo $folder
conda activate dl # Activate python environment for log2plot and the script generating the latent error video
plot_util="$HOME/code/inference_nnimvs/plot" #log2plot python script path
plot_error_util="$HOME/code/aevs/plots/generate_error_plot.py"
launch_dir=`pwd`
generate_latent_error_video=true
generate_log2plot_videos=true
robot_video="dog_v3.mp4"
for f in $folder/*; do
    if [ -d "$f" ]; then
        image_folder="$f/images"
        video_folder="$f/videos"
        mkdir $video_folder
        echo "Creating videos for $f in $video_folder"
        idiff_out="$video_folder/Idiff.mkv"
        i_out="$video_folder/I.mkv"
        latent_error_video="$f/latent_error.mkv"
        latent_error_video_resized="$video_folder/latent_error.mkv"

        if $generate_latent_error_video = true; then
            python $plot_error_util $f
            ffmpeg -y -i "$latent_error_video" -vf scale=320:234 "$latent_error_video_resized"
        fi
        
        ffmpeg -y -i "$image_folder/Idiff%5d.jpg" -pix_fmt yuv420p "$idiff_out" 
        ffmpeg -y -i "$image_folder/I%5d.jpg" -pix_fmt yuv420p "$i_out"
        
        cd $f
        python "$plot_util" visp_pose_nn.yaml  -s 10 --legendLoc 4 --legendCol 2 -s 18 --fontAxes 6 --fontLegend 4 --fontLabel 18 
        python "$plot_util" visp_velocity_nn.yaml --nodisplay -s 10 --legendLoc 4 --legendCol 2 -s 18 --fontAxes 6 --fontLegend 4 --fontLabel 18
        python "$plot_util" visp_pose3D_nn.yaml visp_pose3D_pbvs.yaml visp_pose3D_dvs.yaml --legend 'Siame-se(3)' PBVS DVS  --nbcam 0  -i --fig 5 5 --fontLegend 8 --fontLabel 18 -s 18 
        python "$plot_util" visp_pose3D_nn_full_scene.yaml visp_pose3D_pbvs_full_scene.yaml --nodisplay --legend 'Siame-se(3)' PBVS --nbcam 2
        python "$plot_util" visp_ssd_nn.yaml --nodisplay --legendLoc 2 -s 18 --fontAxes 6 --fontLegend 4 --fontLabel 18 
        python "$plot_util" visp_error_nn.yaml --nodisplay --legendLoc 1 --legendCol 2 -s 18 --fontAxes 6 --fontLegend 4 --fontLabel 18 
        python "$plot_error_util" visp_error_nn.yaml
        if "$generate_log2plot_videos" = true; then
            python "$plot_util" visp_pose_nn.yaml --nodisplay -s 10 --legendLoc 4 --legendCol 2 -s 18 --fontAxes 6 --fontLegend 4 --fontLabel 18
            python "$plot_util" visp_velocity_nn.yaml --nodisplay -s 10 --legendLoc 4 --legendCol 2 -s 18 --fontAxes 6 --fontLegend 4 --fontLabel 18 -v 1
            python "$plot_util" visp_pose3D_nn.yaml visp_pose3D_pbvs.yaml visp_pose3D_dvs.yaml --legend 'Siame-se(3)' PBVS DVS  --nbcam 0  -i --fig 5 5 --fontLegend 8 --fontLabel 18 -s 18 
            python "$plot_util" visp_pose3D_nn_full_scene.yaml visp_pose3D_pbvs_full_scene.yaml --nodisplay --legend 'Siame-se(3)' PBVS --nbcam 2
            python "$plot_util" visp_ssd_nn.yaml --nodisplay --legendLoc 2 -s 18 --fontAxes 6 --fontLegend 4 --fontLabel 18 -v 1
            python "$plot_util" visp_error_nn.yaml --nodisplay --legendLoc 1 --legendCol 2 -s 18 --fontAxes 6 --fontLegend 4 --fontLabel 18 -v 1
            python "$plot_error_util" visp_error_nn.yaml
        fi
        cd ..
        # mv "$f/visp_velocity_nn.mp4" "$video_folder/visp_velocity_nn.mkv"
        # mv "$f/visp_ssd_nn.mp4" "$video_folder/visp_ssd_nn.mkv"
        # mv "$f/visp_error_nn.mp4" "$video_folder/visp_error_nn.mkv"
        velocity_video="$video_folder/velocity_resized.mkv"
        ssd_video="$video_folder/ssd_resized.mkv"
        pose_error_video="$video_folder/pose_error_resized.mkv"
        
        ffmpeg -y -i "$f/visp_velocity_nn.mp4" -vf scale=394:-2 "$velocity_video"
        ffmpeg -y -i "$f/visp_ssd_nn.mp4" -vf scale=320:234 "$ssd_video"
        ffmpeg -y -i "$f/visp_error_nn.mp4" -vf scale=394:-2 "$pose_error_video"
        
        ffmpeg -y -i "$image_folder/I0.jpg" -pix_fmt yuv420p "$video_folder/I0.jpg" # Do this so that the final video is saved in color: ffmpeg uses the pixel format of the first input stream


        ffmpeg -y -i "$video_folder/I0.jpg" -i "$image_folder/Id.jpg" -i "$i_out" \
        -i "$idiff_out" -i "$latent_error_video_resized" -i "$ssd_video" -i "$pose_error_video" -i "$velocity_video" -filter_complex \
        "[0]drawtext=text='I0':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v0];
        [1]drawtext=text='I*':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v1];
        [2]drawtext=text='I(t)':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v2];
        [3]drawtext=text='I(t) - I*':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v3];
        [4]drawtext=text='z(t) - z*':fontsize=16:x=(w - text_w)/2 + w/20:y=h/4:fontcolor=black[v4];
        [5]drawtext=text='||I(t) - I*||²':fontsize=16:x=(w - text_w)/2 + w/20:y=h/4:fontcolor=black[v5];
        [6]drawtext=text='Pose error':fontsize=16:x=(w - text_w)/2 + w/20:y=h/4*3:fontcolor=black[v6];
        [7]drawtext=text='Velocities':fontsize=16:x=(w - text_w)/2 + w/8:y=h/6:fontcolor=black[v7];
        [v0][v1][v2][v3][v4][v5][v6][v7]xstack=inputs=8:layout=0_0|w0_0|0_h0|w0_h0|w0+w1_0|w0+w1_h0|0_h0+h1|w6_h0+h1[v]" -map "[v]" -r 24 -pix_fmt yuv420p "$video_folder/full.mkv"

        # if "$robot_video"; then
        #     $robot_video_cleaned="$video_folder/robot_view.mp4"
        #     ffmpeg -y -i "$f/$robot_video" -vf scale=730:-2 "$robot_video_cleaned"

        # fi
        
        cd $launch_dir

        # $f is a directory
    fi
done
