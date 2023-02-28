#!/bin/bash
# Download/unzip labels
d='../datasets' # unzip directory
url=https://github.com/BowieHsu/EfficientTeacher/releases/tag/data_list/
f='data_list.zip' # 
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
wait # finish background tasks