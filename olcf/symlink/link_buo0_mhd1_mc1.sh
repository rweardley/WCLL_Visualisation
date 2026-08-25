SOURCE_DIR="/lustre/orion/proj-shared/nfu106/dalinger/inputs/lm-mhd/2.1-nekrs/05_m3_mhd_no_buo/11run/"
OUTPUT_DIR="/lustre/orion/fus166/proj-shared/rweb/gb_final/buo0_mhd1_mc1/"

CASENAME="pink5m3_mhd_no_buo"
CHECKPOINT=00034

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "SOURCE_DIR=$SOURCE_DIR" > $OUTPUT_DIR/log.link
echo "CHECKPOINT=$CHECKPOINT" >> $OUTPUT_DIR/log.link

ln -sf $SOURCE_DIR/${CASENAME}0.f00000 $OUTPUT_DIR/${CASENAME}0.f00000
ln -sf $SOURCE_DIR/${CASENAME}0.f${CHECKPOINT} $OUTPUT_DIR/${CASENAME}0.f00001
cp $SOURCE_DIR/${CASENAME}.nek5000 $OUTPUT_DIR
sed -i 's/^numtimesteps:.*/numtimesteps: 2/' $OUTPUT_DIR/${CASENAME}.nek5000

ln -sf $SOURCE_DIR/mhd${CASENAME}0.f00000 $OUTPUT_DIR/mhd${CASENAME}0.f00000
ln -sf $SOURCE_DIR/mhd${CASENAME}0.f${CHECKPOINT} $OUTPUT_DIR/mhd${CASENAME}0.f00001
cp $SOURCE_DIR/mhd${CASENAME}.nek5000 $OUTPUT_DIR
sed -i 's/^numtimesteps:.*/numtimesteps: 2/' $OUTPUT_DIR/mhd${CASENAME}.nek5000

ln -sf $SOURCE_DIR/curr${CASENAME}0.f00000 $OUTPUT_DIR/curr${CASENAME}0.f00000
ln -sf $SOURCE_DIR/curr${CASENAME}0.f${CHECKPOINT} $OUTPUT_DIR/curr${CASENAME}0.f00001
cp $SOURCE_DIR/curr${CASENAME}.nek5000 $OUTPUT_DIR
sed -i 's/^numtimesteps:.*/numtimesteps: 2/' $OUTPUT_DIR/curr${CASENAME}.nek5000