import os
import re
import io
import bioc
from typing import TypedDict, Union

from model_ner import HUGFACE_NER
from processing_data import ml_intext_fn,out_BIO_BERT_crf_fn,out_BIO_BERT_softmax_fn
from restore_index import NN_restore_index_fn
import stanza
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu))
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)


class NERPubtatorResponse(TypedDict):
    start: int
    end: int
    entity: str
    entity_type: str

AIONerResponse = Union[
    NERPubtatorResponse,
    str
]

class ModelSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance.load_model()
        return cls._instance


    def load_model(self):
        self.nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'},package='None') #package='craft'

        self.model = os.getenv('DEFAULT_MODEL')
        self.entity = os.getenv('DEFAULT_ENTITY')
        self.vocabfile = os.getenv('DEFAULT_VOCABFILE')

        model_paras = self.model.split('/')[-1].split('-')
        self.para_set = {
            'encoder_type': model_paras[0].lower(),  # pubmedbert or bioformer
            'decoder_type': model_paras[1].lower(),  # crf or softmax
            'entity_type': self.entity,
            'vocabfile': self.vocabfile
        }
        if self.para_set['encoder_type'] == 'pubmedbert':
            self.vocabfiles = {
                'labelfile': self.para_set['vocabfile'],
                'checkpoint_path': '../pretrained_models/BiomedNLP-PubMedBERT-base-uncased-abstract/',
                'lowercase': True,
            }
        elif self.para_set['encoder_type'] == 'bioformer':
            self.vocabfiles = {
                'labelfile': self.para_set['vocabfile'],
                'checkpoint_path': '../pretrained_models/bioformer-cased-v1.0/',
                'lowercase': False,
            }
        self.nn_model=HUGFACE_NER(self.vocabfiles)
        self.nn_model.build_encoder()
        if self.para_set['decoder_type']=='crf': 
            self.nn_model.build_crf_decoder()
        elif self.para_set['decoder_type']=='softmax':
            self.nn_model.build_softmax_decoder()
            
        self.nn_model.load_model(self.modelfile)
    # input format is either PubTator or BioC. We will usually use PubTator
    def process_content(self, content, input_format):
        processed_data: AIONerResponse = None
        if(input_format == "PubTator"):
            # need to convert pubtator to bioc
            collection = bioc.pubtator2bioc(content)
            processed_data = self.NER_BioC(collection)    

        elif(input_format == "text"):
            processed_data = self.NER_PubTator(content)
        elif(input_format == "BioC"):
            collection = bioc.loads(content)
            processed_data = self.NER_BioC(collection)    
        return processed_data
    def ML_Tag(self, text):
        decoder_type=self.para_set['decoder_type']
        entity_type=self.para_set['entity_type']
        conll_in=self.ssplit_token(text, entity_type, max_len=self.ml_model.maxlen)
        #print(ssplit_token) 
    #    print('ssplit token:',time.time()-startTime)
        
    #    startTime=time.time()
        ml_tsv=ml_tagging(conll_in,self.ml_model,decoder_type=decoder_type)
        #print('ml_tsv:\n',ml_tsv)
    #    print('ml ner:',time.time()-startTime)
    
        final_result= NN_restore_index_fn(text,ml_tsv)
        # print('final ner:',time.time()-startTime)
        
        return final_result
    def pre_token(self,sentence):
        sentence=re.sub("([\=\/\(\)\<\>\+\-\_])"," \\1 ",sentence)
        sentence=re.sub("[ ]+"," ",sentence)
        return sentence

    def ssplit_token(self, in_text,entity_type,max_len=400):
        #print('max_len:',max_len)
        fout=io.StringIO()

        in_text=in_text.strip()
        in_text=self.pre_token(in_text)
        doc_stanza = self.nlp(in_text)
        strlen=0
        for sent in doc_stanza.sentences:
            fout.write('<'+entity_type+'>\tO\n')
            for word in sent.words:
                strlen+=1
                fout.write(word.text+'\tO\n')
                if strlen>=max_len:
                    # print('long sentence:',strlen)
                    fout.write('\n')
                    strlen=0
            fout.write('</'+entity_type+'>\tO\n')
            fout.write('\n')
            strlen=0           
        return fout.getvalue()

def ml_tagging(ml_input,nn_model,decoder_type='crf'):

    test_list = ml_intext_fn(ml_input)
    if decoder_type=='crf': 
        test_x,test_y, test_bert_text_label=nn_model.rep.load_data_hugface(test_list,word_max_len=nn_model.maxlen,label_type='crf')
    elif decoder_type=='softmax': 
        test_x,test_y, test_bert_text_label=nn_model.rep.load_data_hugface(test_list,word_max_len=nn_model.maxlen,label_type='softmax')

    test_pre = nn_model.model.predict(test_x,batch_size=64)
    if decoder_type=='crf':
        test_decode_temp=out_BIO_BERT_crf_fn(test_pre,test_bert_text_label,nn_model.rep.index_2_label)
    elif decoder_type=='softmax':
        test_decode_temp=out_BIO_BERT_softmax_fn(test_pre,test_bert_text_label,nn_model.rep.index_2_label)

    return test_decode_temp


    def NER_PubTator(self, content) -> list[NERPubtatorResponse]:
        tag_result = self.ML_Tag(content)
        entities = []
        for ele in tag_result:
            ent_start = ele[0]
            ent_last = ele[1]
            ent_mention = content[int(ele[0]):int(ele[1])]
            ent_type=ele[2]
            entities.append({
                "start": ent_start,
                "end" : ent_last,
                "entity" : ent_mention,
                "entity_type":ent_type
            })
        return entities
    def NER_BioC(self, collection) -> str:
        Total_n=len(collection.documents)
        print('Total number of sub-documents:', Total_n)
        doc_num=0
        for document in collection.documents:
            print("Processing:{0}%".format(round(doc_num * 100 / Total_n)), end="\r")
            doc_num+=1
            # print(document.id)
            mention_num=0
            for passage in document.passages:
                if passage.text!='' and (not passage.text.isspace()) and passage.infons['type']!='ref': # have text and is not ref
                    passage_offset=passage.offset
                    tag_result=self.ML_Tag(passage.text)
                    for ele in tag_result:
                        bioc_note = bioc.BioCAnnotation()
                        bioc_note.id = str(mention_num)
                        mention_num+=1
                        bioc_note.infons['type'] = ele[2]
                        start = int(ele[0])
                        last = int(ele[1])
                        loc = bioc.BioCLocation(offset=str(passage_offset+start), length= str(last-start))
                        bioc_note.locations.append(loc)
                        bioc_note.text = passage.text[start:last]
                        passage.annotations.append(bioc_note)
        res = bioc.dumps(collection, pretty_print=False)
        return res
