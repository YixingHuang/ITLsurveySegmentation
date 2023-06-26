import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cmd0 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name  SI --runmode first_task_basemodel_dump  --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_glioma   --fixed_init_lr 0.001 --num_class 2 --lr_grid 1e-3 --batch_size 4 --no_maximal_plasticity_search'

cmd1 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name  SI  --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_glioma --fixed_init_lr 0.001 --num_class 2  --lr_grid 1e-3 --batch_size 1 --no_maximal_plasticity_search'

cmd2 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name  joint  --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_glioma --fixed_init_lr 0.001 --num_class 2  --batch_size 1 --lr_grid 1e-3  --no_maximal_plasticity_search'


cmds = [cmd2]
for cmd in cmds:
    os.system(cmd)

