Install Git Push Hook
============================

In your top directory, running `./tools/install-codestyle-check-githook.sh`.
The hook script will be linked to file **.git/hooks/pre-push**.

cpplint will check your codestyle according to google cpp codestyle, if there are error,
the script will exit with 1 which will stop your pushing action. However if you insist on
pushing, run `unlink .git/hooks/pre-push`.
