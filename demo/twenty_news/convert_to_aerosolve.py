#!/usr/bin/python

import os
import re

base_dir = "20_newsgroups"

directory = os.listdir(base_dir)

exclude = re.compile(r"^From:|^To:|^Path:|^Newsgroups:|^Subject:|^Xref:|^Summary:|^Keywords:|^Date:|^Expires:|^Followup-To:|^Message-ID:|^Distribution:|^Organization:|^Supersedes|^Approved:|^Archive-name:|^Lines:|^Sender:|^References:|^Last-modified:|^NNTP-Posting-Host:|^Reply-To:")

output = open("20_newsgroups.txt", "w")

for dir in directory:
  subdir_name = os.path.join(base_dir, dir)
  subdir = os.listdir(subdir_name)
  for filename in subdir:
    fullpath = os.path.join(base_dir,dir, filename)
    file = open(fullpath, "r")
    lines = ""
    
    for line in file:
      if not exclude.match(line):
        lines = lines + line.strip('\n').replace('\t', ' ') + " "
    output.write(dir + "\t" + lines + "\n")
output.close
