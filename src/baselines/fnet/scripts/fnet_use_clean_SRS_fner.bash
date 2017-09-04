dataset_name=SRS_fner
data_directory=../data/processed/baselines/fnet/data/$dataset_name
ckpt_directory=../data/processed/baselines/fnet/fnet_use_clean/$dataset_name

mkdir -p $data_directory
mkdir -p $ckpt_directory

rnn_hidden_neurons=200
keep_prob=0.5
learning_rate=0.002
batch_size=500
char_embedding_size=200
char_rnn_hidden_neurons=50
joint_embedding_size=500
epochs=4
save_checkpoint_after=500000

glove_vector_file_path=/hdd/word_vectors/glove.42B.300d/glove.42B.300d.txt


#echo "Generate local variables required for model"
#python baselines/fnet/src/data_processing/json_to_tfrecord.py prepare_local_variables ../data/datasets/$dataset_name/fner_train.json  $glove_vector_file_path unk $data_directory/ --lowercase

#echo "Converting Train data to TFRecord"
#python baselines/fnet/src/data_processing/json_to_tfrecord.py afet_data $data_directory/ ../data/datasets/$dataset_name/fner_train.json

echo "Converting Test data to TFRecord"
python baselines/fnet/src/data_processing/json_to_tfrecord.py afet_data $data_directory/ ../data/datasets/fner_dev.json --test_data
python baselines/fnet/src/data_processing/json_to_tfrecord.py afet_data $data_directory/ ../data/datasets/fner_eval.json --test_data

# Run train test 5 times
for ((i=1; i<=1; i++)); do
  # Do not emit '_run_' from model ckpt name
  # format: prefix_run_suffix
  model_ckpt_name=200_0.5_0.002_500_200_50_500_run_$i

#  echo "Training a FNET model"
#  time python baselines/fnet/src/main_fnet_train.py  $data_directory/ $ckpt_directory/$model_ckpt_name/ 'fner_train.json_*.tfrecord' $rnn_hidden_neurons $keep_prob $learning_rate $batch_size $char_embedding_size $char_rnn_hidden_neurons $joint_embedding_size $epochs $save_checkpoint_after --use_mention --use_clean

  echo "Testing a FNET model on FNER dev data"
  python baselines/fnet/src/main_fnet_test.py $ckpt_directory/$model_ckpt_name/ $data_directory/fner_dev.json_0.tfrecord

  echo "Testing a FNET model on FNER eval data"
  python baselines/fnet/src/main_fnet_test.py $ckpt_directory/$model_ckpt_name/ $data_directory/fner_eval.json_0.tfrecord


  echo "Report results FNER dev data"
  bash scripts/report_result_fnet.bash $ckpt_directory/$model_ckpt_name/fner_dev.json_0.tfrecord/ ../data/datasets/fner_dev.json 0 > $ckpt_directory/$model_ckpt_name/fner_dev.json_0.tfrecord/final_result.txt

  echo "Report results FNER eval data"
  bash scripts/report_result_fnet.bash $ckpt_directory/$model_ckpt_name/fner_eval.json_0.tfrecord/ ../data/datasets/fner_eval.json 0 > $ckpt_directory/$model_ckpt_name/fner_eval.json_0.tfrecord/final_result.txt

done

