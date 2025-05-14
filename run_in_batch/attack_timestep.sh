# export EXPERIMENT_NAME=$data_id
export EXPERIMENT_NAME="SimAC"
export MODEL_PATH=$model_path
export CLEAN_TRAIN_DIR="$data_path/$dataset_name/$data_id/set_A" 
export CLEAN_ADV_DIR="$data_path/$dataset_name/$data_id/set_B"
export CLEAN_REF="$data_path/$dataset_name/$data_id/set_C"
# export OUTPUT_DIR="outputs/simac/$dataset_name/$EXPERIMENT_NAME"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/$wandb_run_name"
export CLASS_DIR=$class_dir


# ------------------------- Train ASPL on set B -------------------------
mkdir -p $OUTPUT_DIR
rm -r $OUTPUT_DIR/* 2>/dev/null || true
cp -r $CLEAN_REF $OUTPUT_DIR/image_clean_ref
cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

simac_cmd="""accelerate launch attacks/time_feature.py \
--pretrained_model_name_or_path=$MODEL_PATH  \
--enable_xformers_memory_efficient_attention \
--instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
--instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
--instance_prompt='$instance_prompt' \
--class_data_dir=$class_data_dir \
--num_class_images=200 \
--class_prompt='$class_prompt' \
--output_dir=$OUTPUT_DIR \
--center_crop \
--with_prior_preservation \
--prior_loss_weight=1.0 \
--resolution=512 \
--train_text_encoder \
--train_batch_size=1 \
--max_train_steps=$attack_steps \
--max_f_train_steps=3 \
--max_adv_train_steps=6 \
--checkpointing_iterations=$attack_steps \
--learning_rate=5e-7 \
--pgd_alpha=0.005 \
--pgd_eps=$r \
--mixed_precision=$mixed_precision  \
--report_to=$report_to \
--seed=0
"""
echo $simac_cmd
eval $simac_cmd

# export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/$attack_steps"
# export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/simac/CelebA-HQ/$EXPERIMENT_NAME"

# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_PATH  \
#   --enable_xformers_memory_efficient_attention \
#   --train_text_encoder \
#   --instance_data_dir=$INSTANCE_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --instance_prompt="a photo of sks person" \
#   --class_prompt="a photo of person" \
#   --inference_prompt="a photo of sks person" \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-7 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=200 \
#   --max_train_steps=1000 \
#   --checkpointing_steps=1000 \
#   --center_crop \
#   --mixed_precision=bf16 \
#   --prior_generation_precision=bf16 \
#   --sample_batch_size=1 \
#   --seed=0
  
# python infer.py \
#   --model_path $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000 \
#   --output_dir $DREAMBOOTH_OUTPUT_DIR/checkpoint-1000-test-infer


