### Environment Setup
```sh
/home/shardul.junagade/miniconda3/bin/conda create -n thera python=3.10 -y

source /home/shardul.junagade/miniconda3/bin/activate thera
```

pip install --upgrade pip
pip install -r requirements.txt
```


### Remove Environment
```sh
/home/shardul.junagade/miniconda3/bin/conda env remove -n thera
```

Watch GPU Memory Usage
```sh
watch -n 1 "nvidia-smi | awk '/Processes:/ {flag=1; next} flag'"
```

## Super-resolve any image with:
> ./super_resolve.py IN_FILE OUT_FILE --scale 3.14 --checkpoint thera-rdn-pro.pkl

./super_resolve.py "./data/bihar/train/images/9306974_2877087.tif" "./9306974_2877087_out.tif" --scale 4 --checkpoint "thera-rdn-pro.pkl"


## Run on a folder of images:
python run_eval.py --checkpoint thera-rdn-pro.pkl --data-dir path_to_data_parent_folder --eval-sets data_folder_name --save-dir .


### Using my own script:

export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nohup python batch_super_resolve.py "./data/bihar/test/test_bihar_same_class_count_10_120_1000/images" "./results/bihar_thera_rdn_pro/test/" --checkpoint "thera-rdn-pro.pkl" --scale 4 > test.log 2>&1 &