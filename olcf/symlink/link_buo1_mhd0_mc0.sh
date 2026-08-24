SOURCE_DIR="/lustre/orion/nfu106/proj-shared/aldeia/09_GB26_finalist/02_m3/13run/"
OUTPUT_DIR="/lustre/orion/fus166/proj-shared/rweb/gb_final/buo1_mhd0_mc0/"

CASENAME="pink5m3"
CHECKPOINT=00052

echo "SOURCE_DIR=$SOURCE_DIR" > $OUTPUT_DIR/log.link
echo "CHECKPOINT=$CHECKPOINT" >> $OUTPUT_DIR/log.link

ln -sf $SOURCE_DIR/${CASENAME}0.f${CHECKPOINT} $OUTPUT_DIR/${CASENAME}0.f00000
cp $SOURCE_DIR/${CASENAME}.nek5000 $OUTPUT_DIR
sed -i 's/^numtimesteps:.*/numtimesteps: 1/' $OUTPUT_DIR/${CASENAME}.nek5000
