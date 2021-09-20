Repository for MADS 2020 
Bank Marketing

Task: Which person is most likely to take a deposit

Steps:  1) Structure Data
        2) Run models
        3) Cross validate the models
        4) Create presentation


Workflow for git contribution: 
1.)  Select clone with htpps (or ssh)
2.) Clone Repository to your machine
then:
1.) git checkout -b 'yourBranchName'                    (this will be required in the beginning before you make new changes)
2.) before you do something:
        git pull                                            (gets the latest changes on all branches)
3.) after changing stuff:
        git status                                          (shows you which files you changed)
        git add 'fileName'                                  (select all files you want to commit)
        git commit -m 'write an explict change message'     (commits all added files, message will show others what you have done, so try to be preceise)
        git push                                            (this will ask for your username & password)

If errors accure, carefully read the error message

4.) rais a pull request from your branch to the master using gitlab

5.) you can then either start a new branch (as in step 1) or stay in this branch and make new changes

remote fetching:
You can also add a branch from another teammember to your local repositroys by using:
        git checkout remote 'branchname'

