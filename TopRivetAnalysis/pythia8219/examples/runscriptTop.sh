#!/bin/bash 
#o
#e

mkdir /tmp/gunnep/
rm /tmp/gunnep/main42_$SGE_TASK_ID.hepmc
source /nfs/dust/cms/user/gunnep/TopRivetAnalysis/local/rivetenv.sh
export RIVET_ANALYSIS_PATH=/nfs/dust/cms/user/gunnep/TopRivetAnalysis/
export HEPMCOUT=/tmp/gunnep/main42_$SGE_TASK_ID.hepmc
cp main42Top.cmnd main42Top_$SGE_TASK_ID.cmnd
sed -i 's/SEEDPYTHIA8/'$SGE_TASK_ID'/g' main42Top_$SGE_TASK_ID.cmnd
./main42 main42Top_$SGE_TASK_ID.cmnd /tmp/gunnep/main42_$SGE_TASK_ID.hepmc &
rivet -a CMS_TopJets -H outputTop_$SGE_TASK_ID.yoda  $HEPMCOUT > rivet$SGE_TASK_ID.out
rm /tmp/gunnep/main42_$SGE_TASK_ID.hepmc
