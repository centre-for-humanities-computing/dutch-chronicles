# Chronicling Crises: Event Detection in Early Modern Chronicles from the Low Countries

[Paper](nothing.html)

## Abstract
Between the Middle Ages and the nineteenth century, many middle-class Europeans kept a handwritten chronicle, in which they reported on events they considered relevant. Discussed topics varied from records of price fluctuations to local politics, and from weather reports to remarkable gossip. What we do not know yet, is to what extent times of conflict and crises influenced the way in which people dealt with information. We have applied methods from information theory -- dynamics in word usage and measures of relative entropy such as \textit{novelty} and \textit{resonance} -- to a corpus of early modern chronicles from the Low Countries (1500--1820) to provide more insight in the way early modern people were coping with information during impactful events. We detect three peaks in the novelty signal, which coincide with times of political uncertainty in the Northern and Southern Netherlands. Topic distributions provided by Top2Vec show that during these times, chroniclers tend to write more and more extensively about an increased variation of topics. 

<br>

## Project Organization
The organization of the project is as follows:

```
├── lda/                <- trained top2vec model
│   └── ...
├── notebooks/          <- jupyter notebooks with exploratory analyses
│   └── ...
├── output/             <- examples for the paper
│   └── ...
│
├── src/                <- analysis scripts
│   ├── chronicles/         <- reusable
│   │   ├── entropies/          <- calculating indicator variables (incl. novelty)
│   │   ├── misc/               <- handling dates, etc.
│   │   ├── parser/             <- xml parsing, document segmentation
│   │   └── representation/     <- finding prototypes
│   │
|   └── application/         <- ad hoc scripts
│       ├── config/             <- yaml files specifying parameters for experiments
│       ├── topics/             <- training the top2vec model
│       ├── visualization/      <- some more complicated publication plots (the less complicated are in notebooks/)
│       ├── convert_from_xml.sh <- shell script for running XML parsing
│       └── novelty_signal.py   <- pipeline for fitting the novelty signal
│
└── requirements.txt      <- install this 
```

## Usage
Clone, create a virtual environment, install dependencies (assumes you have `virtualenv` installed)
```  
git clone https://github.com/centre-for-humanities-computing/dutch-chronicles.git
cd dutch-chronicles/
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Source code of the experiment is organized into has three main steps:  
    1) parsing XML chronicle files, segment into documents  
    2) train a top2vec model  
    3) fit the novelty signal, which includes prototype picking  

### Parsing XML files
```
cd src/application/
bash convert_from_xml.sh
```

### Train a top2vec model
```
cd src/application/
python topics/top2vec_training.py
```

### Novelty signal
```
cd src/application/
python novelty_signal.py --yml config/220815_prototypes_day.yml
```

Alternatively, you can specify your own config file, following the instructions in [novelty_signal](https://github.com/centre-for-humanities-computing/dutch-chronicles/blob/main/src/application/novelty_signal.py).