import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cmd0 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name  SI --runmode first_task_basemodel_dump  --test  --test_overwrite_mode --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet   --fixed_init_lr 0.001 --num_class 2 --lr_grid 1e-3 --batch_size 4 --no_maximal_plasticity_search'

cmd1 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name  SI  --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_glioma --fixed_init_lr 0.001 --num_class 2  --lr_grid 1e-3 --batch_size 1 --no_maximal_plasticity_search'

cmd2 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name unet_training --method_name  joint  --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 50 --n_iters 1  --first_task_basemodel_folder first_task_glioma --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd3 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training2 --method_name  FT  --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd4 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name  SI  --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd5 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name   meanIMM  --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd6 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name   EWC --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd7 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name   LWF --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd8 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training2 --method_name   LWF --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd9 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training2 --method_name   MAS --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd10 = 'py ../framework/main.py AutoEncoder_cl_128_128  --gridsearch_name base_training2 --method_name  SI --runmode first_task_basemodel_dump --ds_name glioma  --n_tasks  4  --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd11 = 'py ../framework/main.py AutoEncoder_cl_128_128  --gridsearch_name base_training2 --method_name   EBLL --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd12 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name base_training --method_name   modeIMM  --ds_name glioma  --n_tasks  4 --test  --test_overwrite_mode --num_epochs 10 --n_iters 1  --first_task_basemodel_folder first_task_glioma_unet --fixed_init_lr 0.001 --num_class 2  --batch_size 4 --lr_grid 1e-3  --no_maximal_plasticity_search'

cmd13 = 'py ../framework/main.py UNet_cl_128_128  --gridsearch_name GliomaSWT_500redo1 --method_name  FT  --ds_name glioma   --test  --test_overwrite_mode --num_epochs 100 --n_iters 1  --first_task_basemodel_folder  first_task_GliomaSWT_500redo1 --optimizer 1  --fixed_init_lr 0.001 --num_class 2 --batch_size 16  --no_maximal_plasticity_search --stochastic --seed 1'
cmds = [cmd13]
for cmd in cmds:
    os.system(cmd)





