
task=$1
GloVe="data/glove/glove.840B.300d.txt"
SQuAD="data/squad/train-v1.1.json"
QUASART="data/quasart/raw/train_contexts.json.gz"
UnfTriviaqa="data/unftriviaqa/raw/triviaqa-rc.tar.gz"
Searchqa="data/searchqa/raw/SearchQA.zip"

if [ ! -f "$GloVe" ]; then
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/glove
	unzip -o -d data/glove/ data/glove/glove.840B.300d.zip
fi;

if [ "$task" = "quasart" ]; then

	if [ ! -f "$QUASART" ]; then
		wget http://curtis.ml.cmu.edu/datasets/quasar/quasar-t/contexts/short/train_contexts.json.gz -P data/quasart/raw
		wget http://curtis.ml.cmu.edu/datasets/quasar/quasar-t/contexts/short/dev_contexts.json.gz -P data/quasart/raw
		wget http://curtis.ml.cmu.edu/datasets/quasar/quasar-t/contexts/short/test_contexts.json.gz -P data/quasart/raw
		wget http://curtis.ml.cmu.edu/datasets/quasar/quasar-t/questions/train_questions.json.gz -P data/quasart/raw
		wget http://curtis.ml.cmu.edu/datasets/quasar/quasar-t/questions/dev_questions.json.gz -P data/quasart/raw
		wget http://curtis.ml.cmu.edu/datasets/quasar/quasar-t/questions/test_questions.json.gz -P data/quasart/raw
	fi;

    if [ ! -d "data/quasart/sequence" ]; then
		mkdir data/quasart/sequence
	fi;

	python preprocess.py quasart

elif [ "$task" = "unftriviaqa" ]; then
    if [ ! -f "$UnfTriviaqa" ]; then
        wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz -P data/unftriviaqa/raw
        wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz -P data/unftriviaqa/raw
        tar -xf data/unftriviaqa/raw/triviaqa-rc.tar.gz -C data/unftriviaqa/raw/
        tar -xf data/unftriviaqa/raw/triviaqa-unfiltered.tar.gz -C data/unftriviaqa/raw/
    fi;
    if [ ! -d "data/unftriviaqa/sequence" ]; then
		mkdir data/unftriviaqa/sequence
	fi;
    python preprocess.py unftriviaqa

elif [ "$task" = "searchqa" ]; then
    if [ ! -d "data/searchqa" ]; then
		mkdir data/searchqa
	fi;
    if [ ! -d "data/searchqa/raw" ]; then
		mkdir data/searchqa/raw
	fi;
    if [ ! -f "$Searchqa" ]; then
        ggID='0B51lBZ1gs1XTR3BIVTJQWkREQU0'
        ggURL='https://drive.google.com/uc?export=download'
        filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
        getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
        curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "$Searchqa"
        #echo "!!!!!!!!Please dowload the file \"SearchQA.zip\" to the path data/searchqa/raw/ through address: https://drive.google.com/open?id=0B51lBZ1gs1XTR3BIVTJQWkREQU0"
        #exit 1
    fi;
    unzip -o -d data/searchqa/raw $Searchqa
    if [ ! -d "data/searchqa/sequence" ]; then
		mkdir data/searchqa/sequence
	fi;
    python preprocess.py searchqa

elif [ "$task" = "quasartans" ]; then
    if [ ! -d "data/quasartans" ]; then
		mkdir data/quasartans
	fi;
    if [ ! -d "data/quasartans/sequence" ]; then
		mkdir data/quasartans/sequence
	fi;
    if [ ! -f "data/quasartans/vocab.t7" ]; then
        cp data/quasart/*t7 data/quasartans/
        cp data/quasart/sequence/*testing.txt data/quasartans/sequence
        cp trainedmodel/quasart1_best trainedmodel/quasart1_best_backup
    fi;
    python preprocess.py quasartans

elif [ "$task" = "searchqaans" ]; then
    if [ ! -d "data/searchqaans" ]; then
		mkdir data/searchqaans
	fi;
    if [ ! -d "data/searchqaans/sequence" ]; then
		mkdir data/searchqaans/sequence
	fi;
    if [ ! -f "data/searchqaans/vocab.t7" ]; then
        cp data/searchqa/*t7 data/searchqaans/
        cp data/searchqa/sequence/*testing.txt data/searchqaans/sequence
        cp trainedmodel/searchqa1_best trainedmodel/searchqa1_best_backup
    fi;
    python preprocess.py searchqaans

elif [ "$task" = "unftriviaqaans" ]; then
    if [ ! -d "data/unftriviaqaans" ]; then
		mkdir data/unftriviaqaans
	fi;
    if [ ! -d "data/unftriviaqaans/sequence" ]; then
		mkdir data/unftriviaqaans/sequence
	fi;
    if [ ! -f "data/unftriviaqaans/vocab.t7" ]; then
        cp data/unftriviaqa/*t7 data/unftriviaqaans/
        mkdir data/unftriviaqaans/raw
        cp -R data/unftriviaqa/raw/triviaqa-unfiltered/ data/unftriviaqaans/raw
        cp trainedmodel/unftriviaqa1_best trainedmodel/unftriviaqa1_best_backup
    fi;
    python preprocess.py unftriviaqaans

elif [ "$task" = "squad" ]; then
	if [ ! -f "$SQuAD" ]; then
		wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -P data/squad
		wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P data/squad
	fi;
	if [ ! -d "data/squad/sequence" ]; then
		mkdir data/squad/sequence
	fi;
	curl https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ > trainedmodel/evaluation/squad/evaluate-v1.1.py
	python preprocess.py squad
fi;
