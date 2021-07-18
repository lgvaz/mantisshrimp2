cuda_version_major="${1}" 
case ${cuda_version_major} in 
   10)  
      echo "Installing icevision + dependencices for CUDA ${1}"
      echo "- Installing torch and its dependencies"
      echo "- Installing fastai"
      pip install fastai==2.3.1 --upgrade -q
      pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html --upgrade -q
      pip install torchtext==0.9.0 --upgrade -q
      echo "- Installing icevision from master"
      pip install git+git://github.com/airctic/icevision.git\#egg=icevision[all] --upgrade -q
      echo "- Installing icedata from master"      
      pip install git+git://github.com/airctic/icedata.git --upgrade -q
      echo "- Installing mmcv"
      pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html --upgrade -q
      echo "- Installing mmdet"
      pip install mmdet==2.13.0 --upgrade -q
      echo "- Installing yolov5-icevision" 
      pip install yolov5-icevision --upgrade -q 
      echo "icevision installation finished!"   
      ;; 
   11)  
      echo "Installing icevision + dependencices for cuda ${1}"
      echo "- Installing fastai"
      pip install fastai==2.3.1 --upgrade -q
      echo "- Installing torch and its dependencies"
      pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html --upgrade 
      pip install torchtext==0.9.0 --upgrade
      echo "- Installing icevision from master"
      pip install git+git://github.com/airctic/icevision.git\#egg=icevision[all] --upgrade -q
      echo "- Installing icedata from master"      
      pip install git+git://github.com/airctic/icedata.git --upgrade -q
      echo "- Installing mmcv"
      pip install mmcv-full=="1.3.3" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html --upgrade  
      echo "- Installing mmdet"
      pip install mmdet==2.13.0 --upgrade -q
      echo "- Installing yolov5-icevision" 
      pip install yolov5-icevision --upgrade -q 
      echo "icevision installation finished!" 
    ;;       
   *)  
      echo "Coud not install icevision. Check out which torch and torchvision versions are compatible with your CUDA version" 
      exit -1 # Command to come out of the program with status -1
      ;; 
esac 
