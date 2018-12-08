TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
for i in *.cc; do
  echo $i
  g++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared $i -o ${i::-2}so -fPIC -I $TF_INC -L$TF_LIB -ltensorflow_framework
done
