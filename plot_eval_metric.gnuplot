set term png
set output filename.'.png'
set key autotitle columnheader
n = system("awk 'NR==1 {print NF}' '".filename.".tsv'")
plot for [i=2:n] filename.'.tsv' using 1:i with linespoints
