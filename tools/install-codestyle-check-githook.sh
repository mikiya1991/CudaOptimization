#!/bin/sh

HOOKPATH=$(find $PWD -wholename "*.git/hooks" -type d)
echo ${HOOKPATH}

if [ -z ${HOOKPATH} ]; then
    echo "please run in dir with .git"
    exit 1
fi

cd .git/hooks; \
    ln -s ../../tools/cpplint_pre_push_hook.sh pre-push
