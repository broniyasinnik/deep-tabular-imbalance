#find ./Keel1 -name "*.zip" | while read filename; do unzip -o -d "`dirname "$filename"`" "$filename"; done;
#find ./imb_IRlowerThan9 -name "*.dat" -exec sed -i '' -e "/^@/d" {} \;

# Find all files not matching the regex (-o is or -a is and) (! is not)
find ./Keel1 -type f ! -regex ".*1tra.dat" -a ! -regex ".*1tst.dat" | xargs rm

#Change the name of the files in  Keel
for i in {0..4}; do
        find . -type f -name "result$i*.tra" | while read filename; do
        # echo $filename
        DIR=$(dirname "$filename")
        PARNAME=$(basename "$DIR")
        NEW="$DIR/${PARNAME}-5-$((i+1))smt.dat"
        mv $filename $NEW
        done
done