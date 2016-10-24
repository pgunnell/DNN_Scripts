#!/bin/bash 
#o
#e

rm /tmp/gunnep/_$SGE_TASK_ID.hepmc
source /nfs/dust/cms/user/gunnep/TopRivetAnalysis/local/rivetenv.sh
export RIVET_ANALYSIS_PATH=/nfs/dust/cms/user/gunnep/TopRivetAnalysis/
export HEPMCOUT=/tmp/gunnep/_$SGE_TASK_ID.hepmc
cp main42QCD.cmnd main42QCD_$SGE_TASK_ID.cmnd
sed -i 's/SEEDPYTHIA8/'$SGE_TASK_ID'/g' main42QCD_$SGE_TASK_ID.cmnd
./main42 main42QCD_$SGE_TASK_ID.cmnd /tmp/gunnep/_$SGE_TASK_ID.hepmc &
rivet -a CMS_TopJets -H outputQCD_$SGE_TASK_ID.yoda  $HEPMCOUT > rivet$SGE_TASK_ID.out
rm /tmp/gunnep/_$SGE_TASK_ID.hepmc
rm main42QCD_$SGE_TASK_ID.cmnd
