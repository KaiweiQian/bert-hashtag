# bert-hashtag
This is the final project of MSDS 597 at Rutgers University. 

Run the train.py with:
nohup python -u train.py --n_epoch=[n_epoch] 
                         --epoch_per_save=[epoch_per_save] 
                         --max_len=[max_len] 
                         --batch_size=[batch_size] 
                         --max_grad_norm=[max_grad_norm]
                         --lr=[lr]
                         --gamma=[gamma] > model.log &
                         
Run the evaluate.py with:
python evaluate.py --model_file=[model_file] 
                   --eval_file=[eval_file] 
                   --max_len=[max_len] 
                   --batch_size=[batch_size]
