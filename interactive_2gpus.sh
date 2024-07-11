# max memory: 101.3 GB; --mem=256g won't work
srun --partition=haopeng --time=10:00:00 --nodes=1 --gres=gpu:H100:2 --tasks=1 --tasks-per-node=1 --cpus-per-task=32 --mem=96g --pty bash -i

# srun --partition=secondary --time=4:00:00 --nodes=1 --gres=gpu:H100:2 --tasks=1 --tasks-per-node=1 --cpus-per-task=32 --mem=96g --pty bash -i