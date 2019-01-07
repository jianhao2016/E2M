#! /bin/sh
#
# download_files.sh
# Copyright (C) 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.
#


curl --remote-name --remote-header-name --request POST --header 'Content-Type: application/json' --data @$1 'https://api.gdc.cancer.gov/data'
