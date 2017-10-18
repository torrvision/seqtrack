awk -F/ 'NR==1{next} NR==FNR{a[$1"/"$2]=1; next} FNR==1{print; next} !a[$1"/"$2]' otb50.csv otb100.csv >otb_diff.csv
