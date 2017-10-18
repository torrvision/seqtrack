datasets="dtb70 nuspro tc uav123"

for d in $datasets; do
    echo "dataset: ${d}"
    cat "${d}.csv" | tail -n +2 | awk -F/ '{print $1"/"$2}' | uniq | bash shuf.sh >"${d}_vids.txt"
    n="$(cat "${d}_vids.txt" | wc -l)"
    ntrain=$(printf "%.0f" "$(echo "0.9*$n" | bc)")
    # nval=$(($n - $ntrain))
    cat "${d}_vids.txt" | head -n "$ntrain" >"${d}_train_vids.txt"
    cat "${d}_vids.txt" | tail -n "+$(($ntrain+1))" >"${d}_val_vids.txt"
    for s in train val; do
        echo "  set: ${d}_${s}"
        # Filter using awk. Keep headers.
        awk -F/ 'NR==FNR{a[$1"/"$2]=1; next} FNR==1{print; next} a[$1"/"$2]' \
            "${d}_${s}_vids.txt" "${d}.csv" >"${d}_${s}.csv"
    done
    rm *_vids*.txt
done

# Concatenate with header row.
awk 'FNR==1 && NR!=1 {next} {print}' $(echo "$datasets" | sed -e 's/\>/_train.csv/g') >pool636_train.csv
awk 'FNR==1 && NR!=1 {next} {print}' $(echo "$datasets" | sed -e 's/\>/_val.csv/g') >pool636_val.csv
