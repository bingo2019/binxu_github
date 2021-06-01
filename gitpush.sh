
if [ "$1" ];then
   export GIT_PUSH_FILE_NAME=$1
else
   echo "Please add the file name. eg. sh gitpush test.py"
fi
git add $GIT_PUSH_FILE_NAME
git commit -m "A file will push to github master breach "
git push origin master


