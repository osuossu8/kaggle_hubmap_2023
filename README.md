# CompetitionBase

# $B;vA0$K?7$7$$%j%]%8%H%j$r(BGitHub$B$G:n@.$7$F$*$/(B

# $B8E$$%j%]%8%H%j$r%/%m!<%s$9$k(B
$ git clone https://github.com/xxx/old-repository.git
$ cd old-repository

# $B8E$$%j%]%8%H%j$N(BGit$B$N99?7>pJs$r:o=|$9$k(B
$ rm -rf .git

# $B?7$7$$%j%]%8%H%j$r?75,:n@.$9$k(B
$ git init
$ git add .
$ git commit -m "Initial commit"
$ git remote add origin https://github.com/xxx/new-repository.git
$ git push -u origin master
