folder=$1
echo $folder
launch_dir=`pwd`
for f in $folder/*; do
    if [ -d "$f" ]; then
        video_folder="$f/videos"
        mkdir $video_folder
        echo "Creating videos for $f in $video_folder"
        idiff_original_out="$video_folder/Idiff_o.mkv"
        i_original_out="$video_folder/I_o.mkv"
        idiff_decoded_out="$video_folder/Idiff_dec.mkv"
        i_decoded_out="$video_folder/I_dec.mkv"
        
        ffmpeg -y -i "$f/orig_Idiff_%5d.jpg" "$idiff_original_out"
        ffmpeg -y -i "$f/orig_%5d.jpg" "$i_original_out"
        ffmpeg -y -i "$f/rec_Idiff_%5d.jpg" "$idiff_decoded_out"
        ffmpeg -y -i "$f/rec_%5d.jpg" "$i_decoded_out"
        
        ffmpeg -y -i "$f/orig_00000.jpg" -i "$f/orig_Id.jpg" -i "$f/rec_00000.jpg" -i "$f/rec_Id.jpg" \
        -i "$i_original_out" -i "$idiff_original_out" -i "$i_decoded_out" -i "$idiff_decoded_out" -filter_complex \
        "[0]drawtext=text='I0':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v0];
        [1]drawtext=text='I*':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v1];
        [2]drawtext=text='~I0':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v2];
        [3]drawtext=text='~I*':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v3];
        [4]drawtext=text='I(t)':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v4];
        [5]drawtext=text='I* - I(t)':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v5];
        [6]drawtext=text='~I(t)':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v6];
        [7]drawtext=text='~I* - ~I(t)':fontsize=20:x=(w-text_w)/2:y=(h-text_h):fontcolor=white[v7];
        [v0][v1][v2][v3][v4][v5][v6][v7]xstack=inputs=8:layout=0_0|w0_0|w0+w1_0|w0+w1+w2_0|0_h0|w0_h0|w0+w1_h0|w0+w1+w2_h0[v]" -map "[v]" "$video_folder/full.mkv"
        cd $f
        #python "$launch_dir/plot_error.py" visp_error_nn.yaml
        cd $launch_dir

        # $f is a directory
    fi
done