gsname='gs1'
echo "RSYNC GRIDSEARCH DATA FROM:"
echo ${gsname}
echo
rsync -r -vam --progress abeukers@scotty.princeton.edu:/jukebox/norman/abeukers/sem/cswsem/toycsw/gsdata/${gsname}/* /Users/abeukers/wd/toycsw/gsdata/${gsname}