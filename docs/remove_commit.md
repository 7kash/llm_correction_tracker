# Removing a Commit on GitHub

Sometimes you push an unwanted commit to a GitHub repository and need to undo it. Here are the two most common workflows.

## Option 1: Revert the commit
Use this when the commit is already published and other people may have pulled it. This creates a new commit that undoes the changes, preserving history.

```bash
git checkout <branch>
git pull origin <branch>
git revert <commit-sha>
git push origin <branch>
```

## Option 2: Rewrite history (reset)
Only use this if the commit is on your own branch and you are comfortable rewriting history. This moves the branch pointer backwards and requires a force-push.

```bash
git checkout <branch>
git pull origin <branch>
git reset --hard <target-commit-sha>
git push --force origin <branch>
```

### Notes
- Replace `<branch>` with the branch name, and `<commit-sha>` with the hash of the commit you want to undo.
- After a revert or reset, confirm your branch state with `git status` and run your test suite if needed.
- Never force-push shared branches such as `main` without coordinating with collaborators.
