# Visual Analytics Assignments

Welcome to the Visual Analytics course. All assignments of the course will be done through this git repository. Each assignment will be its own branch.

On each of the assignment folders, there is a README.md. Take a look on it for more information.

## PIP install and dependencies
To set up your environment, we provided a requirements.txt with all required dependencies. Do NOT use any other dependencies other than the ones listed there.

## How to get the assignment repository
You already should have access to a repository in the following format:
`https://git.cs.dal.ca/courses/2020-fall/csci-6612/<your csid>`

Open git bash or a terminal in your machine, and run: `git clone <the above url>` to get the repository to your machine. If you have ssh configured, you may also run `git clone git@git.cs.dal.ca:courses/2020-fall/csci-6612/<your csid>.git`. Only you, the TAs and the Professor have access to your repository. This is you private gateway to get and submit your assignments, as well as to ask questions during office hours.

## How to handle the branches in the repository
You are configured as a `Developer` in the repository, therefore you are only able to push to unprotected branches. There will be one protected branch for each assignment. You may create any number of branches you may wish.

Every time a new assigment is posted, a new protected branch will appear in the repository. You should do a `git merge` from the contents of that branch into your development branch (which may be `master` or some other of your choice), do the assignments and when you wish to submit it, you will merge your development branch back to the assignment branch, which will accept merge requests until the submission deadline. If anyone needs help to understand this, feel free to ask on the assignments MS teams channel.

## How to submit your assignments
The submission will be made though each of the assignments branch. The branch will be open to merge requests (or even commits, though for your own good practice, prefer only to do merge requests to it) in the period where we accept submissions. At the end of this period, the branch will be blocked for any further commits or merges and the latest commit/merge will be considered your official submission for that assignment. 