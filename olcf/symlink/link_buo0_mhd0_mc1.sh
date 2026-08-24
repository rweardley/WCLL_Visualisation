SOURCE_DIR="/lustre/orion/nfu106/proj-shared/gottems/GB26_finalist/03_m3_no_buo/03_m3_no_buo/31run/"
OUTPUT_DIR="/lustre/orion/fus166/proj-shared/rweb/gb_final/buo0_mhd0_mc1/"

CASENAME="pink5m3_no_buo"
CHECKPOINT=00058

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "SOURCE_DIR=$SOURCE_DIR" > $OUTPUT_DIR/log.link
echo "CHECKPOINT=$CHECKPOINT" >> $OUTPUT_DIR/log.link

ln -sf $SOURCE_DIR/${CASENAME}0.f${CHECKPOINT} $OUTPUT_DIR/${CASENAME}0.f00000
cp $SOURCE_DIR/${CASENAME}.nek5000 $OUTPUT_DIR
sed -i 's/^numtimesteps:.*/numtimesteps: 1/' $OUTPUT_DIR/${CASENAME}.nek5000
