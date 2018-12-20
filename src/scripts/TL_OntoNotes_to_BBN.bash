dataset_name=BBN
data_directory=../data/processed/data/$dataset_name
ckpt_directory=../data/processed/ckpts/$dataset_name

mkdir -p $data_directory
mkdir -p $ckpt_directory


# Update the parameters --tl_ckpt_directory,  --tl_model_number and --tl_local_variables for appropriate model selection related to transfer learning. There parameters are used in step 1 and model training section of this file.

# Model parameters
rnn_hidden_neurons=100
keep_prob=0.5
learning_rate=0.0005
batch_size=800
char_embedding_size=200
char_rnn_hidden_neurons=200
joint_embedding_size=500
epochs=10
save_checkpoint_after=219634

# http://nlp.stanford.edu/data/glove.840B.300d.zip
glove_vector_file_path=../data/glove.840B.300d.txt


# Step 1: Generate local variables such as word to number dictionary etc.
echo "Generate local variables required for model"
python data_processing/json_to_tfrecord.py prepare_local_variables ../data/$dataset_name/train.json  $glove_vector_file_path unk $data_directory/ --transfer_learning --tl_local_variables=../data/processed/data/OntoNotes/local_variables.pickle

# Step 2: Convert Training data into TFRecord format.
echo "Converting Train data to TFRecord"
python data_processing/json_to_tfrecord.py afet_data $data_directory/ ../data/$dataset_name/train.json

# Step 3: Convert development and testing data into TFRecord format.
echo "Converting Dev and Test data to TFRecord"
python data_processing/json_to_tfrecord.py afet_data $data_directory/ ../data/$dataset_name/dev.json --test_data
python data_processing/json_to_tfrecord.py afet_data $data_directory/ ../data/$dataset_name/test.json --test_data
# Run train test procedure 5 times
for ((i=1; i<=1; i++)); do
  # Do not emit '_run_' from model ckpt name
  # format: prefix_run_suffix
  model_ckpt_name=checkpoint_run_$i

  # On GTX 1080 GPU it takes around 6 minutes to train.
  echo "Training a FNET model"
  time python main_fnet_train.py  $data_directory/ $ckpt_directory/$model_ckpt_name/ 'train.json_*.tfrecord' $rnn_hidden_neurons $keep_prob $learning_rate $batch_size $char_embedding_size $char_rnn_hidden_neurons $joint_embedding_size $epochs $save_checkpoint_after --use_mention --use_clean --transfer_learning --tl_ckpt_directory=../data/processed/ckpts/OntoNotes/checkpoint_run_1/ --tl_model_number=43200

  # Testing can take around 10 minutes since it has lot of overhead to initialize models with different checkpoints.
  echo "Testing a FNET model on dev data"
  python main_fnet_test.py $ckpt_directory/$model_ckpt_name/ $data_directory/dev.json_0.tfrecord --hierarchical_prediction

  echo "Testing a FNET model on test data"
  python main_fnet_test.py $ckpt_directory/$model_ckpt_name/ $data_directory/test.json_0.tfrecord --hierarchical_prediction

  # The final_result file contains the result on the development set based on the strict, macro and micro F1 metrics.
  echo "Report results FNER dev data"
  bash scripts/report_result_fnet.bash $ckpt_directory/$model_ckpt_name/dev.json_0.tfrecord/ ../data/$dataset_name/dev.json 0 > $ckpt_directory/$model_ckpt_name/dev.json_0.tfrecord/final_result.txt

  # The final_result file contains the result on the test set based on the strict, macro and micro F1 metrics.
  echo "Report results FNER eval data"
  bash scripts/report_result_fnet.bash $ckpt_directory/$model_ckpt_name/test.json_0.tfrecord/ ../data/$dataset_name/test.json 0 > $ckpt_directory/$model_ckpt_name/test.json_0.tfrecord/final_result.txt

done

