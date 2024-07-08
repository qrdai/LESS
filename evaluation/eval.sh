base_path="/projects/illinois/eng/cs/haopeng/qirundai"

set_save_dir() {
    mdir=$1
    if [[ -d $mdir ]]; then
        # save_dir=${mdir}/eval/$2
        save_dir=${mdir}/$2
    else
        # save_dir=$n/space10/out/$(basename $mdir)/eval/$2
        save_dir=${base_path}/eval_results/${mdir}/$2
    fi
}

# set_valid_dir() {
#     mdir=$1
#     if [[ -d $mdir ]]; then
#         # save_dir=${mdir}/valid/$2
#         save_dir=${mdir}/$2
#     else
#         # save_dir=$n/space10/out/$(basename $mdir)/valid/$2
#         save_dir=${base_path}/valid_results/${mdir}/$2
#     fi
# }

# export DATA_DIR=$n/space10/data/eval
export DATA_DIR="${base_path}/data/eval"
export set_save_dir
# export set_valid_dir

