# -*- coding: utf-8 -*
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import gzip
import json
import pdb
import operator
import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def prepQUASART():

    file_contexts = ['data/quasart/raw/dev_contexts.json.gz', 'data/quasart/raw/test_contexts.json.gz', 'data/quasart/raw/train_contexts.json.gz']
    file_questions = ['data/quasart/raw/dev_questions.json.gz', 'data/quasart/raw/test_questions.json.gz', 'data/quasart/raw/train_questions.json.gz']

    filenames_w = ['data/quasart/sequence/dev.tsv', 'data/quasart/sequence/test.tsv', 'data/quasart/sequence/train.tsv']
    file_testing = ['data/quasart/sequence/dev_testing.txt', 'data/quasart/sequence/test_testing.txt']

    for i in range(3):
        count = 0
        fpr_c = gzip.open(file_contexts[i], 'r')
        fpr_q = gzip.open(file_questions[i], 'r')
        fpw = open(filenames_w[i], 'w')
        if i != 2:
            fpw_testing = open(file_testing[i], 'w')
        for line in fpr_c:
            context = json.loads(line)
            question = json.loads(fpr_q.readline())
            assert(context["uid"] == question["uid"])
            fpw.write(question["question"]+'*split_sign*'+question["uid"]+'*split_sign*'+question["answer"]+'\n')
            for c in context['contexts']:
                fpw.write(c[1]+'*split_sign*'+str(c[0])+'\n')
            fpw.write('*new_instance*\n')
            if i != 2:
                fpw_testing.write(question["uid"]+'\t'+question["answer"]+'\n')

        fpr_c.close()
        fpr_q.close()
        fpw.close()

        if i != 2:
            fpw_testing.close()

    print ('QUASART preprossing finished!')

def prepQuasartans():
    file_contexts = ['data/quasart/raw/dev_contexts.json.gz', 'data/quasart/raw/test_contexts.json.gz', 'data/quasart/raw/train_contexts.json.gz']
    file_questions = ['data/quasart/raw/dev_questions.json.gz', 'data/quasart/raw/test_questions.json.gz', 'data/quasart/raw/train_questions.json.gz']

    file_answers = ['trainedmodel/evaluation/quasart/dev_output_top.txt_backup', 'trainedmodel/evaluation/quasart/test_output_top.txt_backup', 'trainedmodel/evaluation/quasart/train4test_top.txt']

    filenames_w = ['data/quasartans/sequence/dev.tsv', 'data/quasartans/sequence/test.tsv', 'data/quasartans/sequence/train.tsv']

    for i in range(3):
        count = 0
        fpr_c = gzip.open(file_contexts[i], 'r')
        fpr_q = gzip.open(file_questions[i], 'r')

        id_answer = {}
        id_answer2 = {}

        fpr_a = open(file_answers[i], 'r')
        accuracy = 0.0
        accuracy2 = 0.0
        accuracy3 = 0.0
        all_num = 0.0
        ans_num_dict = {}
        for line in fpr_a:
            divs = line.strip().split('\t')
            ans = []
            ans_dict = {}
            max_num = 0
            max_ans = ''
            for j in range(1, len(divs), 2):
                divs[j] = divs[j].lower()#divs[j] = normalize_answer(divs[j].lower())
                if divs[j] not in ans:
                    ans.append(divs[j])
                if divs[j] in ans_dict:
                    ans_dict[divs[j]] += 1.0
                else:
                    ans_dict[divs[j]] = 1.0
                if ans_dict[divs[j]] > max_num:
                    max_num = ans_dict[divs[j]]
                    max_ans = divs[j]

            for k, v in ans_dict.items():
                ans_dict[k] /= ( (len(divs)-1)/2.0 )

            ans_sorted = sorted(ans_dict.items(), key=operator.itemgetter(1), reverse=True)

            for j in range(len(ans)):
                ans_dict[ans_sorted[j][0]] = 1.0 / (j+1)

            top_num = len(ans)

            if top_num > 20:
                top_num = 20
            ans = [ans[j] for j in range(top_num)]

            '''
            max_num = 0
            for a in ans:
                if ans_dict[a] > max_num:
                    max_num = ans_dict[a]
                    max_ans = a
            '''
            id_answer[divs[0]] = [ans, ans_dict]

            #id_answer[divs[0]] = [ans[0], ans_dict]
            #id_answer[divs[0]] = [[max_ans], ans_dict]
        fpr_a.close()
        fpw = open(filenames_w[i], 'w')

        for line in fpr_c:

            context = json.loads(line)
            question = json.loads(fpr_q.readline())
            answer = question["answer"].lower()#normalize_answer(question["answer"].lower())

            assert(context["uid"] == question["uid"])
            if question["uid"] in id_answer:
                all_num += 1
                cand_ans = id_answer[ question["uid"] ][0]
                cand_ans_num = id_answer[ question["uid"] ][1]

                if answer in cand_ans:
                    accuracy += 1

                if i == 2 and answer not in cand_ans:
                    cand_ans.append(answer)
                    cand_ans_num[answer] = cand_ans_num[cand_ans[0]]

                answer_context = {}
                for c in context['contexts']:
                    context = c[1].lower()
                    for cand in cand_ans:
                        if cand in context:
                            if cand in answer_context:
                                answer_context[cand].append(context)
                            else:
                                answer_context[cand] = [ context ]

                if i == 2:
                    if answer in answer_context:
                        fpw.write(question["question"]+'*split_sign*'+question["uid"]+'*split_sign*'+question["answer"]+'\n')
                        for cand in cand_ans:
                            if cand in answer_context:
                                assert(cand in cand_ans_num)
                                fpw.write(cand+'*split_sign*'+str(cand_ans_num[cand])+'*split_sign*'+'*split_sign*'.join(answer_context[cand])+'\n')

                        fpw.write('*new_instance*\n')
                else:
                    if len( answer_context.keys() ) != 0:
                        fpw.write(question["question"]+'*split_sign*'+question["uid"]+'*split_sign*'+question["answer"]+'\n')
                        tmp = False
                        for cand in cand_ans:
                            if cand in answer_context:
                                fpw.write(cand+'*split_sign*'+str(cand_ans_num[cand])+'*split_sign*'+'*split_sign*'.join(answer_context[cand])+'\n')
                                tmp = True
                        assert(tmp == True)


                        fpw.write('*new_instance*\n')
                    else:
                        accuracy2 += 1


        fpr_c.close()
        fpr_q.close()
        fpw.close()
        print(accuracy / all_num)
        print(accuracy2 / all_num)
        print(all_num)

    print ('QUASART Search preprossing finished!')

def prepSearchqa():
    print('Preprossing dataset Searchqa! ')
    filenames = ['data/searchqa/raw/SearchQA/val.txt', 'data/searchqa/raw/SearchQA/test.txt', 'data/searchqa/raw/SearchQA/train.txt']
    filenames_w = ['data/searchqa/sequence/dev.tsv', 'data/searchqa/sequence/test.tsv', 'data/searchqa/sequence/train.tsv']
    filenames_w_testing = ['data/searchqa/sequence/dev_testing.txt', 'data/searchqa/sequence/test_testing.txt', 'data/searchqa/sequence/train_testing.txt']
    question_id = 1
    for i in range(3):
        fpr = open(filenames[i], 'r')
        fpw = open(filenames_w[i], 'w')
        fpw_testing = open(filenames_w_testing[i], 'w')
        for line in fpr:
            con = line.strip().split('|||')
            question = con[1].strip()
            answer = con[2].strip()
            if len(question) != 0 and len(answer) != 0:
                assert(len(question) > 0)
                assert(len(answer) > 0)
                passages = con[0].strip()[4:-4].split('</s>  <s>')
                assert(len(passages) != 0)
                fpw.write(question + '*split_sign*' + str(question_id) + '*split_sign*' + answer + '\n')
                for p in passages:
                    assert(len(p)>0)
                    fpw.write(p + '*split_sign*' + '0\n')
                fpw.write('*new_instance*\n')
                fpw_testing.write( str(question_id) + '\t' + answer + '\n' )
                question_id += 1

        print(question_id)
        fpw_testing.close()
        fpr.close()
        fpw.close()

def prepSearchqaans():

    file_answers = ['trainedmodel/evaluation/searchqa/dev_output_top.txt_backup', 'trainedmodel/evaluation/searchqa/test_output_top.txt_backup', 'trainedmodel/evaluation/searchqa/train4test_top.txt',]
    filenames_w = ['data/searchqaans/sequence/dev.tsv', 'data/searchqaans/sequence/test.tsv', 'data/searchqaans/sequence/train.tsv']
    filenames_r = ['data/searchqa/sequence/dev.tsv', 'data/searchqa/sequence/test.tsv', 'data/searchqa/sequence/train.tsv']

    for i in range(3):
        count = 0


        id_answer = {}
        id_answer2 = {}


        accuracy = 0.0
        accuracy2 = 0.0
        accuracy3 = 0.0
        all_num = 0.0
        pred_id = 0
        ans_num_dict = {}
        fpr_a = open(file_answers[i], 'r')
        for line in fpr_a:
            divs = line.strip().split('\t')
            ans = []
            ans_dict = {}
            max_num = 0
            max_ans = ''
            for j in range(1, len(divs)):
                divs[j] = divs[j].lower()
                if divs[j] not in ans:
                    ans.append(divs[j])
                if divs[j] in ans_dict:
                    ans_dict[divs[j]] += 1
                else:
                    ans_dict[divs[j]] = 1
                if ans_dict[divs[j]] > max_num:
                    max_num = ans_dict[divs[j]]
                    max_ans = divs[j]

            ans = ans[:20]

            id_answer[divs[0]] = ans
            ans_num_dict[divs[0]] = ans_dict
            #id_answer[divs[0]] = [ans[0]]
            #id_answer[divs[0]] = [max_ans]

        fpr_a.close()
        fpw = open(filenames_w[i], 'w')

        fpr = open(filenames_r[i], 'r')
        answer_context = {}
        write_bool = True
        for line in fpr:
            line = line.rstrip()
            if line == '*new_instance*':
                if i == 2:
                    if answer in answer_context and write_bool:
                        fpw.write(question+'*split_sign*'+question_id+'*split_sign*'+answer+'\n')
                        for cand in cand_ans:
                            if cand in answer_context:
                                if cand in ans_num_dict[question_id]:
                                    score = str(ans_num_dict[question_id][cand])
                                else:
                                    score = '0'
                                fpw.write(cand+'*split_sign*'+score+'*split_sign*'+'*split_sign*'.join(answer_context[cand])+'\n')
                        fpw.write('*new_instance*\n')

                else:
                    if len( answer_context.keys() ) != 0 and write_bool:
                        fpw.write(question+'*split_sign*'+question_id+'*split_sign*'+answer+'\n')
                        tmp = False
                        for cand in cand_ans:
                            if cand in answer_context:
                                fpw.write(cand+'*split_sign*'+str(ans_num_dict[question_id][cand])+'*split_sign*'+'*split_sign*'.join(answer_context[cand])+'\n')
                                tmp = True
                        assert(tmp == True)

                        fpw.write('*new_instance*\n')
                    else:
                        accuracy2 += 1

                answer_context = {}
                write_bool = True
            else:
                divs = line.split('*split_sign*')

                if len(divs) == 2:
                    context = divs[0]
                    for cand in cand_ans:
                        if cand in line:
                            if cand in answer_context:
                                answer_context[cand].append(context)
                            else:
                                answer_context[cand] = [ context ]

                elif len(divs) == 3:
                    question = divs[0]
                    question_id = divs[1]
                    answer = divs[2]
                    all_num += 1

                    cand_ans = id_answer[ question_id ]
                    if answer in cand_ans:
                        accuracy += 1

                    if i == 2 and answer not in cand_ans:
                        cand_ans.append(answer)
                else:
                    print(divs)
                    write_bool = False

        fpr.close()
        fpr_a.close()
        fpw.close()
        print(accuracy / all_num)
        print(all_num)

    print ('Searchqa reranking preprossing finished!')

def prepTriviaqa(dataname='unfiltered-web', task='unftriviaqa'):

    reload(sys)
    sys.setdefaultencoding('ISO-8859-1')
    import json
    import glob
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from difflib import SequenceMatcher
    import codecs

    stops = set(stopwords.words('english'))
    print("The preprossing will take a long time ...")
    tt = sent_tokenize('testing tokenizer!')

    path_data = 'data/unftriviaqa/raw/'
    path_qa = path_data + 'triviaqa-unfiltered/'
    filenames = [path_qa+dataname+'-dev.json', path_qa+dataname+'-test-without-answers.json', path_qa+dataname+'-train.json']

    filenames_w = ['data/'+task+'/sequence/dev.tsv', 'data/'+task+'/sequence/test.tsv', 'data/'+task+'/sequence/train.tsv']

    evident_files = glob.glob(path_data+'evidence/'+dataname+'/*.txt')
    evident_files = [ e[45:] for e in evident_files]

    real_num = 0.0
    for i in range(3):
        #print(i)
        evidence_num = 0.0
        all_num = 0.0

        fpr = codecs.open(filenames[i], 'r',encoding='ISO-8859-1', errors='ignore')
        fpw = open(filenames_w[i], 'w')
        data_all = json.loads(fpr.read())["Data"]

        for data in data_all:
            all_num += 1
            question = data["Question"]
            question_id = data["QuestionId"]
            if i != 1 :
                answers = data["Answer"]["Aliases"]
            else:
                answers = ['answer']
            if dataname == 'web' or dataname == 'unfiltered-web':
                pages = data["SearchResults"]
                pages_new = []
                for p_id in range(len(pages)):
                    if "Filename" in pages[p_id]:
                        pages[p_id]["Filename"] = 'web/'+pages[p_id]["Filename"]
                        pages_new.append(pages[p_id])

                pages = data["EntityPages"]
                for p_id in range(len(pages)):
                    if "Filename" in pages[p_id]:
                        pages[p_id]["Filename"] = 'wikipedia/'+pages[p_id]["Filename"].decode('ISO-8859-1','ignore').encode("ISO-8859-1")
                        pages_new.append(pages[p_id])

                pages = pages_new
                evidence_num += len(pages)

            else:
                pages_ent = data["EntityPages"]
                for p_id in range(len(pages_ent)):
                    pages_ent[p_id]["Filename"] = 'wikipedia/'+pages_ent[p_id]["Filename"].decode('ISO-8859-1','ignore').encode("ISO-8859-1")

                pages = pages_ent
            if len(pages) == 0:
                #print(data)
                continue


            context_sents = []
            for page in pages:
                fpr_p = open(path_data+'evidence/'+page["Filename"], 'r')

                for line in fpr_p.readlines():
                    try:
                        context_sents += sent_tokenize(line)
                    except:
                        #print(line)
                        continue

                fpr_p.close()
            context_sents = [s.replace('\n', ' ') for s in context_sents if len(s.strip()) > 4]

            if len(context_sents) == 0:
                continue
            scores = []
            question_tokens = normalize_answer(question).split(' ')
            for sent in context_sents:
                score = 0
                words = normalize_answer(sent).split(' ')
                for token in question_tokens:
                    if token in words and token not in stops:
                        score += 1
                scores.append((sent,score))
            scores = sorted(scores, key=operator.itemgetter(1), reverse=True)
            try:
                fpw.write( ' '.join( word_tokenize(question) ) + '*split_sign*' + question_id + '*split_sign*' + '*split_answer*'.join(answers) + '\n')
                real_num += 1
                if real_num % 1000 == 0:
                    print(str(int(real_num)) + ' / 110000')
                    #print(str(all_num)+' / 11')
            except:
                continue
            passage_num = len(scores)
            if passage_num > 100:
                passage_num = 100

            for j in range(passage_num):
                word_seq = scores[j][0]
                try:
                    fpw.write(' '.join( word_tokenize(word_seq) ) + '*split_sign*' + str(scores[j][1]) + '\n' )
                except:
                    fpw.write(' '.join( normalize_answer(word_seq).split(' ') ) + '*split_sign*' + str(scores[j][1]) + '\n' )
                    #print('word: ' + str(scores[j][1]))
                    continue
            fpw.write('*new_instance*\n')
        fpr.close()
        fpw.close()

        #print(evidence_num / real_num)
        #print(real_num)
        #print(all_num)

def prepTriviaqaans(dataname='unfiltered-web', task='unftriviaqaans'):

    import sys
    reload(sys)
    sys.setdefaultencoding('ISO-8859-1')
    import json
    import glob
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from difflib import SequenceMatcher

    stops = set(stopwords.words('english'))
    path_data = 'data/unftriviaqa/raw/triviaqa-unfiltered/'
    filenames = [path_data+dataname+'-dev.json', path_data+dataname+'-test-without-answers.json', path_data+dataname+'-train.json']

    fpr = open(filenames[2], 'r')
    data_all = json.loads(fpr.read())["Data"]
    search_dict = {}
    for data in data_all:
        question_id = data["QuestionId"]
        if "MatchedWikiEntityName" not in data["Answer"]:
            answer_single = data["Answer"]["Aliases"][0]
        else:
            answer_single = data["Answer"]["MatchedWikiEntityName"]
            assert(answer_single in data["Answer"]["Aliases"])
        assert(question_id not in search_dict)
        search_dict[question_id] = answer_single
    fpr.close()

    file_answers = ['trainedmodel/evaluation/'+task+'/dev_output_top.txt_backup', 'trainedmodel/evaluation/'+task+'/test_output_top.txt_backup', 'trainedmodel/evaluation/'+task+'/train4test_top.txt']
    filenames_r = ['data/'+task+'/sequence/dev.tsv', 'data/'+task+'/sequence/test.tsv', 'data/'+task+'/sequence/train.tsv']
    filenames_w = ['data/'+task+'ans/sequence/dev.tsv', 'data/'+task+'ans/sequence/test.tsv', 'data/'+task+'ans/sequence/train.tsv']


    for i in range(3):
        count = 0

        id_answer = {}
        id_answer2 = {}

        fpr_a = open(file_answers[i], 'r')
        accuracy = 0.0
        accuracy2 = 0.0
        accuracy3 = 0.0
        all_num = 0.0
        ans_num_dict = {}
        for line in fpr_a:
            divs = line.strip().split('\t')
            ans = []
            ans_dict = {}
            max_num = 0
            max_ans = ''
            for j in range(1, len(divs), 2):
                divs[j] = divs[j]
                if divs[j] not in ans:
                    ans.append(divs[j])
                if divs[j] in ans_dict:
                    ans_dict[divs[j]] += 1.0
                else:
                    ans_dict[divs[j]] = 1.0
                if ans_dict[divs[j]] > max_num:
                    max_num = ans_dict[divs[j]]
                    max_ans = divs[j]

            for k, v in ans_dict.items():
                ans_dict[k] /= ( (len(divs)-1)/2.0 )

            ans_sorted = sorted(ans_dict.items(), key=operator.itemgetter(1), reverse=True)



            top_num = len(ans)

            if top_num > 15:
                top_num = 15

            ans = [ans[j] for j in range(top_num)]

            id_answer[divs[0]] = [ans, ans_dict]

        print(len(id_answer.keys()))
        fpr_a.close()

        fpw = open(filenames_w[i], 'w')

        fpr = open(filenames_r[i], 'r')
        answer_context = {}
        write_bool = True
        question_bool = True
        for line in fpr:
            line = line.rstrip('\n')
            if line == '*new_instance*':
                if i == 2:
                    ans_cont_bool = False
                    for k, v in answer_context.items():
                        if k.lower() in answers:
                            ans_cont_bool = True
                            break
                    if ans_cont_bool and write_bool:
                        fpw.write(question+'*split_sign*'+question_id+'*split_sign*'+'*split_answer*'.join(answers)+'\n')
                        for cand in cand_ans:
                            if cand in answer_context:
                                fpw.write(cand+'*split_sign*'+'0'+'*split_sign*'+'*split_sign*'.join(answer_context[cand])+'\n')
                        fpw.write('*new_instance*\n')

                else:
                    if len( answer_context.keys() ) != 0 and write_bool:
                        fpw.write(question+'*split_sign*'+question_id+'*split_sign*'+'*split_answer*'.join(answers)+'\n')
                        tmp = False
                        for cand in cand_ans:
                            if cand in answer_context:
                                fpw.write(cand+'*split_sign*'+'0'+'*split_sign*'+'*split_sign*'.join(answer_context[cand])+'\n')
                                tmp = True
                        assert(tmp == True)

                        fpw.write('*new_instance*\n')
                    else:
                        accuracy2 += 1

                answer_context = {}
                write_bool = True
                question_bool = True
            else:
                divs = line.split('*split_sign*')
                if len(divs) != 2 and len(divs) != 3:
                    print(divs)
                    print(len(divs))
                if len(divs) == 2:
                    divs[0] = divs[0].decode('ISO-8859-1','ignore').encode("ISO-8859-1")
                    context = divs[0].lower()
                    for cand in cand_ans:
                        cand_low = cand.lower()
                        if cand_low in context:
                            if cand in answer_context:
                                answer_context[cand].append(divs[0])
                            else:
                                answer_context[cand] = [ divs[0] ]

                elif len(divs) == 3:
                    assert( question_bool )
                    question = divs[0]
                    question_id = divs[1]

                    answers = divs[2].split('*split_answer*')
                    for k in range(len(answers)):
                        answers[k] = answers[k].lower()
                    all_num += 1
                    if question_id not in id_answer:
                        write_bool = False
                        print(question_id)
                        cand_ans = []
                    else:
                        cand_ans = id_answer[ question_id ][0]
                    ans_cont_bool = False

                    for answer in cand_ans:
                        if answer.lower() in answers:
                            accuracy += 1
                            ans_cont_bool = True
                            break

                    if i == 2 and not ans_cont_bool:
                        answer_single = search_dict[question_id]
                        cand_ans.append(answer_single)
                    question_bool = False
                else:
                    print(divs)
                    write_bool = False






        fpr.close()
        fpr_a.close()
        fpw.close()
        print(accuracy / all_num)

        print(accuracy2 / all_num)
        print(all_num)


def prepSQuAD():
    reload(sys)
    sys.setdefaultencoding('utf-8')
    import json
    from nltk.tokenize import word_tokenize
    count = 0
    filenames = ['dev', 'train']
    for filename in filenames:
        fpr = open("data/squad/"+filename+"-v1.1.json", 'r')
        line = fpr.readline()
        js = json.loads(line)
        fpw = open("data/squad/sequence/"+filename+".txt", 'w')
        for c in js["data"]:
            for p in c["paragraphs"]:
                context = p["context"].split(' ')
                context_char = list(p["context"])
                context_pos = {}
                for qa in p["qas"]:

                    question = word_tokenize(qa["question"])

                    if filename == 'train':
                        for a in qa['answers']:
                            answer = a['text'].strip()
                            answer_start = int(a['answer_start'])

                        #add '.' here, just because NLTK is not good enough in some cases
                        answer_words = word_tokenize(answer+'.')
                        if answer_words[-1] == '.':
                            answer_words = answer_words[:-1]
                        else:
                            answer_words = word_tokenize(answer)

                        prev_context_words = word_tokenize( p["context"][0:answer_start ] )
                        left_context_words = word_tokenize( p["context"][answer_start:] )
                        answer_reproduce = []
                        for i in range(len(answer_words)):
                            if i < len(left_context_words):
                                w = left_context_words[i]
                                answer_reproduce.append(w)
                        join_a = ' '.join(answer_words)
                        join_ar = ' '.join(answer_reproduce)

                        #if not ((join_ar in join_a) or (join_a in join_ar)):
                        if join_a != join_ar:
                            #print join_ar
                            #print join_a
                            #print 'answer:'+answer
                            count += 1

                        fpw.write(' '.join(prev_context_words+left_context_words)+'\t')
                        fpw.write(' '.join(question)+'\t')
                        #fpw.write(join_a+'\t')

                        pos_list = []
                        for i in range(len(answer_words)):
                            if i < len(left_context_words):
                                pos_list.append(str(len(prev_context_words)+i+1))
                        if len(pos_list) == 0:
                            print join_ar
                            print join_a
                            print 'answer:'+answer
                        assert(len(pos_list) > 0)
                        fpw.write(' '.join(pos_list)+'\n')
                    else:
                        fpw.write(' '.join(word_tokenize( p["context"]) )+'\t')
                        fpw.write(' '.join(question)+'\n')

        fpw.close()
    print ('SQuAD preprossing finished!')

def prepSQuADSearch():
    reload(sys)
    sys.setdefaultencoding('utf-8')
    import json
    from nltk.tokenize import word_tokenize, sent_tokenize

    count = 0
    filenames = ['dev', 'train']
    for filename in filenames:
        fpr = open("data/squadans/"+filename+"-v1.1.json", 'r')
        fpr_ans = open("trainedmodel/evaluation/squad/" + filename + "_output_top.txt1", 'r')
        line = fpr.readline()
        js = json.loads(line)
        fpw = open("data/squadans/sequence/"+filename+".tsv", 'w')

        for c in js["data"]:
            for p in c["paragraphs"]:
                context = ' '.join( word_tokenize( p["context"] ) )
                context_sents = sent_tokenize( p["context"] )
                for qa in p["qas"]:

                    question = ' '.join( word_tokenize(qa["question"]) )

                    ans_score = fpr_ans.readline().strip().split('\t')
                    ans = [ans_score[i] for i in range(0,20,2)]
                    scores = [ans_score[i] for i in range(1,20,2)]
                    if filename == 'train':
                        assert(len(qa['answers']) == 1)

                        answer = ' '.join( word_tokenize(qa['answers'][0]['text']) )
                        if answer not in ans:
                            ans.append(answer)
                            scores.append(scores[0])
                        fpw.write(question + '\t' + qa["id"] + '\t' + answer + '\n')
                        for i in range(len(ans)):
                            fpw.write(ans[i] + '\t' + scores[i] + '\t' + context + '\n')
                        fpw.write('\n')

                    else:
                        fpw.write(question + '\t' + qa["id"] + '\t' + 'answer' + '\n')
                        for i in range(len(ans)):
                            fpw.write(ans[i] + '\t' + scores[i] + '\t' + context + '\n')
                        fpw.write('\n')

        fpw.close()
    print ('SQuAD search preprossing finished!')

if __name__ == "__main__":
    task = sys.argv[1]
    if task == "quasart":
        prepQUASART()
    elif task == "searchqa":
        prepSearchqa()
    elif task == "unftriviaqa":
        prepTriviaqa('unfiltered-web', "unftriviaqa")
    elif task == "squad":
        prepSQuAD()
    elif task == 'quasartans':
        prepQuasartans()
    elif task == 'searchqaans':
        prepSearchqaans()
    elif task == 'unftriviaqaans':
        prepTriviaqaans()
    else:
        print('the task not supported yet')
