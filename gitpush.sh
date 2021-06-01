
if [ "$1" ];then
   export GIT_PUSH_FILE_NAME=$1
else
   echo "Please add the file name. eg. sh gitpush test.py"
fi
git add $GIT_PUSH_FILE_NAME
git commit -m "A file will push to github master breach "

if [ "$2" ];then
   export GIT_PUSH_TO_CLOUD=$2
   git push origin master
else
   echo "git success"

