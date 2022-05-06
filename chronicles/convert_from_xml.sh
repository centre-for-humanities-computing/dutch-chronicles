###
### INPUT PARAMS: change as you please!
###
dir_corpus_annotated='../../chronicling-topics/corpus/corpus_220503/corpus_220503_annotated'
dir_corpus_corrected='../../chronicling-topics/corpus/corpus_220503/corpus_220503_corrected'
run_when='220503'


###
### SETUP: does output dir exist? 
###
if [ -d "../../chronicling-topics/primitives_${run_when}" ] 
then
    echo "Directory ${wdir} exists." 
else
    mkdir -p "../../chronicling-topics/primitives_${run_when}"
    echo "Creating ${run_when} output dir in ../data/"
fi


###
### RUNNING
###

echo "Parsing corpus_annotated"
python parser/xml_parsing.py \
    -d "${dir_corpus_annotated}" \
    -s 1 \
    -o "../../chronicling-topics/primitives_${run_when}/primitives_annotated.ndjson"

echo "Parsing corpus_corrected"
python parser/xml_parsing.py \
    -d "${dir_corpus_corrected}" \
    -s 1 \
    -o "../../chronicling-topics/primitives_${run_when}/primitives_corrected.ndjson"

echo "Generating IDs"
python parser/give_ids.py \
    -ap "../../chronicling-topics/primitives_${run_when}/primitives_annotated.ndjson" \
    -cp "../../chronicling-topics/primitives_${run_when}/primitives_corrected.ndjson"

echo "Cleaning date tags on corrected corpus"
python misc/date_tag_resolutions.py \
    -i "../../chronicling-topics/primitives_${run_when}/primitives_corrected.ndjson" \
    -o "../../chronicling-topics/primitives_${run_when}/primitives_corrected_daily.ndjson"

echo "Cleaning date tags on annotated corpus"
python misc/date_tag_resolutions.py \
    -i "../../chronicling-topics/primitives_${run_when}/primitives_annotated.ndjson" \
    -o "../../chronicling-topics/primitives_${run_when}/primitives_annotated_daily.ndjson"