# CompetitionBase

# 事前に新しいリポジトリをGitHubで作成しておく

# 古いリポジトリをクローンする
$ git clone https://github.com/xxx/old-repository.git
$ cd old-repository

# 古いリポジトリのGitの更新情報を削除する
$ rm -rf .git

# 新しいリポジトリを新規作成する
$ git init
$ git add .
$ git commit -m "Initial commit"
$ git remote add origin https://github.com/xxx/new-repository.git
$ git push -u origin master
