module load tools/rclone/1.68.1

rclone copy -P sciebo:Data/textClust_stream_data/embedded ./embedded
rclone copy -P sciebo:Data/textClust_stream_data/raw_data ./raw_data
rclone copy -P sciebo:Data/textClust_stream_data/translated_no_emojis ./translated_no_emojis