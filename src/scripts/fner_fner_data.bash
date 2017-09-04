data_directory=../data/processed/joint_model/data/fner
ckpt_directory=../data/processed/joint_model/ckpt
model_ckpt_name=fner_data_2_layers_

mkdir -p $data_directory
mkdir -p $ckpt_directory

#rm -rf $ckpt_directory/$model_ckpt_name

rnn_hidden_neurons=100
keep_prob=0.5
learning_rate=0.005
batch_size=500
char_embedding_size=100
char_rnn_hidden_neurons=50
joint_embedding_size=500
epochs=6
save_checkpoint_after=10000

glove_vector_file_path=/hdd/word_vectors/glove.42B.300d/glove.42B.300d.txt

#echo "Convert ConLL data to json"
#python utils/conll_to_json.py ../data/raw/conll/eng.testa $data_directory/eng.testa.json

#echo "Generate local variables required for model"
#python json_to_tfrecord.py prepare_local_variables ../data/processed/wikification/fner_train.json $glove_vector_file_path unk $data_directory/ 30 --lowercase

#echo "Converting Train data to TFRecord"
#python json_to_tfrecord.py json_data $data_directory/ ../data/processed/wikification/fner_train.json

#echo "Converting Test data to TFRecord"
#python json_to_tfrecord.py json_data $data_directory/ ../data/raw/figer_gold.json
#python json_to_tfrecord.py json_data $data_directory/ ../data/processed/wikification/fner_test.json --test_data
python json_to_tfrecord.py json_data $data_directory/ $data_directory/eng.testa.json --test_data

#echo "Training a FNET model"
#time python train.py  $data_directory/ $ckpt_directory/$model_ckpt_name/ 'fner_train.json_*.tfrecord' $rnn_hidden_neurons $keep_prob $learning_rate $batch_size $char_embedding_size $joint_embedding_size $epochs $save_checkpoint_after --use_char_cnn

#echo "Testing a FNET model"
#python test.py $ckpt_directory/$model_ckpt_name/ $data_directory/figer_gold.json_0.tfrecord
#python test.py $ckpt_directory/$model_ckpt_name/ $data_directory/fner_test.json_0.tfrecord

#echo "Report results FIGER data."
#bash scripts/report_result_fnet.bash $ckpt_directory/$model_ckpt_name/figer_gold.json_0.tfrecord/ ../data/raw/figer_gold.json 0 ../data/raw/label_patch_prediction_only.txt

#echo "Report results FNER data."
#bash scripts/report_result_fnet.bash $ckpt_directory/$model_ckpt_name/fner_test.json_0.tfrecord/ ../data/processed/wikification/fner_test.json 0

#echo "Results FIGER data NED only"
#bash scripts/report_result.bash ../utils/conlleval.txt $ckpt_directory/$model_ckpt_name/figer_gold.json_0.tfrecord_ned/

#echo "Results FNER data NED only"
#bash scripts/report_result.bash ../utils/conlleval.txt $ckpt_directory/$model_ckpt_name/fner_test.json_0.tfrecord_ned/
