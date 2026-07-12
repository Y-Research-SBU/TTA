#!/bin/bash

gpuid=$1
config=$2
data_prefix=${3:-${TTA_DATA_ROOT:-../../data}}

### Dataset Information

declare -a dataroots=(
	"${data_prefix}/tcga_blca_uni"/
)

task='BLCA_uni_survival'
target_col='dss_survival_days'
split_names='train,val,test'

split_dir='survival/TCGA_BLCA_overall_survival_k=0'
bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
split_dir='survival/TCGA_BLCA_overall_survival_k=1'
bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
split_dir='survival/TCGA_BLCA_overall_survival_k=2'
bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
split_dir='survival/TCGA_BLCA_overall_survival_k=3'
bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
split_dir='survival/TCGA_BLCA_overall_survival_k=4'
bash "./scripts/survival/${config}.sh" $gpuid $task $target_col $split_dir $split_names "${dataroots[@]}"
