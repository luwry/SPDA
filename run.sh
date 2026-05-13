model_name="gpt-3.5-turbo"
###cluster_demos
for num_clusters in 4 10; do
  python cluster_demos.py \
    --pred_file demos/${model_name}_zero_shot_cot/ \
    --demo_save_dir cluster_demos/${model_name}_${num_clusters}/ \
    --num_clusters ${num_clusters}
###iter_demos
for iter in 0 4; do
  python iter_demos_max_demo.py \
    --pred_file 'cluster_demos/gpt-3.5-turbo' \
    --demo_save_dir 'iter_demos/gpt-3.5-turbo' \
    --iter ${iter}
done


## argument role prediction
python inference.py \
  --pred_file iter_demos/gpt-3.5-turbo_4_0/ \
  --output_file results/gpt-3.5-turbo_4_0/
python evaluate.py  \
  --output_file results/gpt-3.5-turbo_4_0/ \
  --num_pred_roles 5

##argument extaction
python inference.py \
  --pred_file iter_demos/gpt-3.5-turbo_10_4/ \
  --output_file results/gpt-3.5-turbo_10_4/
python evaluate.py  \
  --output_file results/gpt-3.5-turbo_10_4/ \
  --num_pred_roles 6