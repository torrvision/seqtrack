set term png
set output filename.'.png'
set key autotitle columnheader
# Determine number of columns.
n = system("awk 'NR==1 {print NF}' '".filename.".tsv'")
# The first column is the step number.
# After that, every alternate column gives the standard error.
plot for [i=2:n:2] filename.'.tsv' using 1:i:(1.96*column(i+1)) with errorbars
# plot for [i=2:n:2] filename.'.tsv' using 1:i:(1.96*column(i+1)) with errorbars, \
#      for [i=2:n:2] filename.'.tsv' using 1:i with lines notitle ls (i/2)
