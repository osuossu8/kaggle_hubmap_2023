# CompetitionBase

## https://docs.github.com/ja/repositories/creating-and-managing-repositories/duplicating-a-repository

・create repo on github

$ git clone --bare https://github.com/EXAMPLE-USER/OLD-REPOSITORY.git

$ cd OLD-REPOSITORY.git

$ git push --mirror https://github.com/EXAMPLE-USER/NEW-REPOSITORY.git

$ cd ..

$ rm -rf OLD-REPOSITORY.git
