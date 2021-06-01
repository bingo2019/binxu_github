if [ "$1" ];then
   export GIT_PUSH_FILE_NAME=$1
else
   echo "Please check the file name. eg. sh gitpush test.py"
fi
git rm -r $GIT_PUSH_FILE_NAME
git commit -m "A file will be deleted in github master commit"
git push
