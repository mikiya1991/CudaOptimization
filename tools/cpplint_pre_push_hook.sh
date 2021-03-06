#!/bin/sh
#
# Modified from http://qiita.com/janus_wel/items/cfc6914d6b7b8bf185b6
#
# An example hook script to verify what is about to be committed.
# Called by "git commit" with no arguments.  The hook should
# exit with non-zero status after issuing an appropriate message if
# it wants to stop the commit.
#
# To enable this hook, rename this file to "pre-commit".

if git rev-parse --verify HEAD >/dev/null 2>&1
then
    against=HEAD
else
    # Initial commit: diff against an empty tree object
    against=4b825dc642cb6eb9a060e54bf8d69288fbee4904
fi

# Redirect output to stderr.
exec 1>&2

cpplint=./tools/cpplint.py
sum=0
filters='-build/include_order,-build/namespaces,-legal/copyright,-runtime/references'

# for cpp
for file in $(git diff-index --name-status $against -- | grep -E '\.[ch]((pp)|u)?$' | awk '{print $2}'); do
    $cpplint --linelength=120 --filter=$filters $file
    sum=$(expr ${sum} + $?)
done

if [ ${sum} -eq 0 ]; then
    exit 0
else
    exit 1
fi
